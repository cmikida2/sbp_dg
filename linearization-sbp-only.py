# Copyright (C) 2020 Cory Mikida
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This routine creates a linearized RHS operator, using the same
# advection test case. It does so by using perturbations/FD.


import numpy as np
import pyopencl as cl  # noqa
import pyopencl.array as parray # noqa
import pyopencl.clmath  # noqa

import pytest  # noqa

from pyopencl.tools import (  # noqa
                pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import numpy.linalg as la
from operators import (sbp21, sbp42, sbp63)
from scipy import sparse
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)


# Domain: x = [-1,1], y = [-1,1]
# SBP Subdomain: x = [-1,0], y = [0,1] (structured mesh)
# DG Subdomain: x = [0,1], y = [0,1] (unstructured mesh)

def main(write_output=True, order=4):

    # Set up the problem in the same way as before.
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    nelem_x = 10
    nelem_y = 10

    dt_factor = 4
    h = 1/40

    c = np.array([0.1, 0.1])
    norm_c = la.norm(c)

    final_time = 0.2

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    # SBP Half.

    # First, need to set up the structured mesh.
    n_sbp_x = nelem_x
    n_sbp_y = nelem_y*2
    x_sbp = np.linspace(-1, 0, n_sbp_x, endpoint=True)
    y_sbp = np.linspace(-1, 1, n_sbp_y, endpoint=True)
    dx = x_sbp[1] - x_sbp[0]
    dy = y_sbp[1] - y_sbp[0]

    # Set up solution vector:
    # For now, timestep is the same as DG.
    u_sbp = parray.zeros(queue, int(n_sbp_x*n_sbp_y), np.float64)

    # Initial condition
    for j in range(0, n_sbp_y):
        for i in range(0, n_sbp_x):
            u_sbp[i + j*(n_sbp_x)] = np.sin(10*(-c.dot(
                                                [x_sbp[i], y_sbp[j]])/norm_c))

    # obtain P and Q
    order_sbp = 4
    if order_sbp == 2:
        [p_x, q_x] = sbp21(n_sbp_x)
        [p_y, q_y] = sbp21(n_sbp_y)
    elif order_sbp == 4:
        [p_x, q_x] = sbp42(n_sbp_x)
        [p_y, q_y] = sbp42(n_sbp_y)
    elif order_sbp == 6:
        [p_x, q_x] = sbp63(n_sbp_x)
        [p_y, q_y] = sbp63(n_sbp_y)

    tau_l = 1
    tau_r = 1

    # for the boundaries
    el_x = np.zeros(n_sbp_x)
    er_x = np.zeros(n_sbp_x)
    el_x[0] = 1
    er_x[n_sbp_x-1] = 1
    e_l_matx = np.zeros((n_sbp_x, n_sbp_x,))
    e_r_matx = np.zeros((n_sbp_x, n_sbp_x,))

    for i in range(0, n_sbp_x):
        for j in range(0, n_sbp_x):
            e_l_matx[i, j] = el_x[i]*el_x[j]
            e_r_matx[i, j] = er_x[i]*er_x[j]

    el_y = np.zeros(n_sbp_y)
    er_y = np.zeros(n_sbp_y)
    el_y[0] = 1
    er_y[n_sbp_y-1] = 1
    e_l_maty = np.zeros((n_sbp_y, n_sbp_y,))
    e_r_maty = np.zeros((n_sbp_y, n_sbp_y,))

    for i in range(0, n_sbp_y):
        for j in range(0, n_sbp_y):
            e_l_maty[i, j] = el_y[i]*el_y[j]
            e_r_maty[i, j] = er_y[i]*er_y[j]

    # construct the spatial operators
    d_x = np.linalg.inv(dx*p_x).dot(q_x - 0.5*e_l_matx + 0.5*e_r_matx)
    d_y = np.linalg.inv(dy*p_y).dot(q_y - 0.5*e_l_maty + 0.5*e_r_maty)

    # for the boundaries
    c_l_x = np.kron(tau_l, (np.linalg.inv(dx*p_x).dot(el_x)))
    c_r_x = np.kron(tau_r, (np.linalg.inv(dx*p_x).dot(er_x)))
    c_l_y = np.kron(tau_l, (np.linalg.inv(dy*p_y).dot(el_y)))
    c_r_y = np.kron(tau_r, (np.linalg.inv(dy*p_y).dot(er_y)))

    # For speed...
    dudx_mat = -np.kron(np.eye(n_sbp_y), d_x)
    dudy_mat = -np.kron(d_y, np.eye(n_sbp_x))

    def rhs(t, u):
        # Initialize the entire RHS to 0.
        rhs_out = parray.zeros(queue,
                               int(n_sbp_x*n_sbp_y), np.float64)

        # Fill the first part with the SBP half of the domain.

        # Pull the SBP vector out of device array for now.
        u_sbp_ts = u.get()

        dudx = np.zeros((n_sbp_x*n_sbp_y))
        dudy = np.zeros((n_sbp_x*n_sbp_y))

        dudx = dudx_mat.dot(u_sbp_ts)
        dudy = dudy_mat.dot(u_sbp_ts)

        # Boundary condition
        dl_x = np.zeros(n_sbp_x*n_sbp_y)
        dr_x = np.zeros(n_sbp_x*n_sbp_y)
        dl_y = np.zeros(n_sbp_x*n_sbp_y)
        dr_y = np.zeros(n_sbp_x*n_sbp_y)

        # Need to fill this by looping through each segment.
        # X-boundary conditions:
        for j in range(0, n_sbp_y):
            u_bcx = u_sbp_ts[j*n_sbp_x:((j+1)*n_sbp_x)]
            v_l_x = np.transpose(el_x).dot(u_bcx)
            v_r_x = np.transpose(er_x).dot(u_bcx)
            left_bcx = np.sin(10*(-c.dot(
                                  [x_sbp[0], y_sbp[j]])/norm_c + norm_c*t))
            right_bcx = np.sin(10*(-c.dot(
                                   [x_sbp[n_sbp_x-1],
                                    y_sbp[j]])/norm_c + norm_c*t))
            dl_xbc = c_l_x*(v_l_x - left_bcx)
            dr_xbc = c_r_x*(v_r_x - right_bcx)
            dl_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dl_xbc
            dr_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dr_xbc
        # Y-boundary conditions:
        for i in range(0, n_sbp_x):
            u_bcy = u_sbp_ts[i::n_sbp_x]
            v_l_y = np.transpose(el_y).dot(u_bcy)
            v_r_y = np.transpose(er_y).dot(u_bcy)
            left_bcy = np.sin(10*(-c.dot(
                                [x_sbp[i], y_sbp[0]])/norm_c + norm_c*t))
            right_bcy = np.sin(10*(-c.dot(
                                [x_sbp[i],
                                 y_sbp[n_sbp_y-1]])/norm_c + norm_c*t))
            dl_ybc = c_l_y*(v_l_y - left_bcy)
            dr_ybc = c_r_y*(v_r_y - right_bcy)
            dl_y[i::n_sbp_x] = dl_ybc
            dr_y[i::n_sbp_x] = dr_ybc

        # Add these at each point on the SBP half to get the SBP RHS.
        rhs_sbp = c[0]*dudx + c[1]*dudy - dl_x - dr_x - dl_y - dr_y

        # Now pop this back into the device RHS vector.
        rhs_sbp_dev = parray.zeros(queue, (n_sbp_x*n_sbp_y,), np.float64)
        rhs_sbp_dev = parray.to_device(queue, rhs_sbp)

        rhs_out = rhs_sbp_dev

        return rhs_out

    # Now, instead of timestepping, we'll make
    # ourselves a massive RHS operator,
    # and (hopefully) find its eigenvalues.

    # Make a combined u with the SBP and the DG parts.
    # Total pts.
    npoints = int(n_sbp_x*n_sbp_y)
    u_base = parray.zeros(queue, npoints, np.float64)
    u_base = u_sbp

    # Get base RHS
    base_rhs = rhs(0, u_base)

    # Perturbation.
    eps = 1e-9

    # Our beautiful matrix.
    rhs_op_dense = np.zeros((npoints, npoints))
    rhs_op = sparse.csc_matrix(rhs_op_dense)
    rhs_op = rhs_op.tolil()
    print('Initialized rhs operator')
    print('Dimension: ', npoints)
    print('Entering point loop')

    for i in range(0, npoints):

        print('Calculating RHS for perturbed point ', i)
        u_pert = u_base.copy()
        u_pert[i] += eps
        pert_rhs = rhs(0, u_pert)

        # RHS diff...
        pert_vec = (pert_rhs - base_rhs) / eps

        # Makes a column of the array.
        # FIXME: there has to be a smarter way than looping here
        for j in np.nonzero(pert_vec)[0]:
            rhs_op[j, i] = pert_vec[j].get()

    print('RHS Op Constructed: nnz = ', rhs_op.count_nonzero())
    # Pop out a spy just to see what the operator
    # looks like.
    plt.spy(rhs_op)
    plt.title('Nonzeros of RHS Operator for SBP-DG')
    plt.savefig("op_spy.png")

    # To save, convert to CSC format.
    rhs_op_csc = rhs_op.tocsc()
    sparse.save_npz("{npoints}x{npoints}_sbp_only.npz".format(npoints=npoints),
                    rhs_op_csc)


if __name__ == "__main__":
    main()
