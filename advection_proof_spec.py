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


import numpy as np
import pyopencl as cl  # noqa
import pyopencl.array as parray # noqa
import pyopencl.clmath  # noqa

import pytest  # noqa

from pyopencl.tools import (  # noqa
                pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import numpy.linalg as la
from operators import (sbp21, sbp42, sbp63)
from projection import sbp_sbp_projection
from scipy import sparse
from matplotlib import pyplot as plt

import logging
logger = logging.getLogger(__name__)


# Domain: x = [-1,0], y = [-1,1]
# SBP Subdomain: x = [-1,0], y = [-1,1] (structured mesh)

def main(write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dt_factor = 10
    h = 1/40

    c = np.array([0.1, 0])
    norm_c = la.norm(c)

    def f(x):
        return sym.sin(10*x)

    def u_analytic(x):
        return f(-c.dot(x)/norm_c+sym.var("t", sym.DD_SCALAR)*norm_c)

    final_time = 1.0

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    # SBP Half.

    # First, need to set up the structured mesh.
    n_sbp_x = 20
    n_sbp_y = 40
    x_sbp = np.linspace(-np.pi/2, 0, n_sbp_x, endpoint=True)
    y_sbp = np.linspace(-np.pi/2, np.pi/2, n_sbp_y, endpoint=True)
    dx = x_sbp[1] - x_sbp[0]

    # Set up solution vector:
    # For now, timestep is the same as DG.
    u_sbp = parray.zeros(queue, int(n_sbp_x*n_sbp_y), np.float64)

    # Initial condition
    for j in range(0, n_sbp_y):
        for i in range(0, n_sbp_x):
            u_sbp[i + j*(n_sbp_x)] = np.sin(10*(-c.dot(
                                                [y_sbp[j], x_sbp[i]])/norm_c))

    # obtain P and Q
    order_sbp = 4
    if order_sbp == 2:
        [p_x, q_x] = sbp21(n_sbp_x)
    elif order_sbp == 4:
        [p_x, q_x] = sbp42(n_sbp_x)
    elif order_sbp == 6:
        [p_x, q_x] = sbp63(n_sbp_x)

    # We are trying to prove if this is the only valid tau-value.
    tau_l = 0.5

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

    # construct the spatial operators
    d_x = np.linalg.inv(dx*p_x).dot(q_x - 0.5*e_l_matx + 0.5*e_r_matx)

    # for the boundaries
    c_l_x = np.kron(tau_l, (np.linalg.inv(dx*p_x).dot(el_x)))

    # For speed...
    dudx_mat = -np.kron(np.eye(n_sbp_y), d_x)

    # FIXME: need SBP east to SBP west projection
    east2west, west2east = sbp_sbp_projection(n_sbp_y-1, order_sbp,
                                              n_sbp_y-1, order_sbp)

    def rhs(t, u):
        # Initialize the entire RHS to 0.
        rhs_out = parray.zeros(queue,
                               int(n_sbp_x*n_sbp_y), np.float64)

        # Fill the first part with the SBP half of the domain.

        # Pull the SBP vector out of device array for now.
        u_sbp_ts = u[0:int(n_sbp_x*n_sbp_y)].get()

        dudx = np.zeros((n_sbp_x*n_sbp_y))

        dudx = dudx_mat.dot(u_sbp_ts)

        # Boundary condition
        dl_x = np.zeros(n_sbp_x*n_sbp_y)

        # Pull SBP domain values off of east face.
        # These will be used for the west (left) BC.
        sbp_east = np.zeros(n_sbp_y)
        counter = 0
        for i in range(0, n_sbp_x*n_sbp_y):
            if i == n_sbp_x - 1:
                sbp_east[counter] = u_sbp_ts[i]
                counter += 1
            elif i % n_sbp_x == n_sbp_x - 1:
                sbp_east[counter] = u_sbp_ts[i]
                counter += 1

        # FIXME: Projection from SBP east to SBP west will be included later.
        # sbp_east = east2west.dot(sbp_east)

        # Need to fill this by looping through each segment.
        # X-boundary conditions:
        for j in range(0, n_sbp_y):
            u_bcx = u_sbp_ts[j*n_sbp_x:((j+1)*n_sbp_x)]
            v_l_x = np.transpose(el_x).dot(u_bcx)
            left_bcx = sbp_east[j]
            dl_xbc = c_l_x*(v_l_x - left_bcx)
            dl_x[j*n_sbp_x:((j+1)*n_sbp_x)] = dl_xbc

        # Add these at each point on the SBP half to get the SBP RHS.
        rhs_sbp = c[0]*dudx - dl_x

        # Now pop this back into the device RHS vector.
        rhs_sbp_dev = parray.zeros(queue, (n_sbp_x*n_sbp_y,), np.float64)
        rhs_sbp_dev = parray.to_device(queue, rhs_sbp)

        rhs_out[0:int(n_sbp_x*n_sbp_y)] = rhs_sbp_dev

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

    for i in range(0, npoints):

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
    plt.title('Nonzeros of RHS Operator for SBP Advection')
    plt.savefig("advec_op_spy.png")

    # To save, convert to CSC format.
    rhs_op_csc = rhs_op.tocsc()
    sparse.save_npz(
            "{npoints}x{npoints}_sbp_advec.npz".format(npoints=npoints),
            rhs_op_csc)


if __name__ == "__main__":
    main()
