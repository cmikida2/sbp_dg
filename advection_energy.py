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
import sys
import pyopencl as cl  # noqa
import pyopencl.array as parray # noqa
import pyopencl.clmath  # noqa
import logging

import pytest  # noqa

from pyopencl.tools import (  # noqa
                pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import numpy.linalg as la
from operators import (sbp21, sbp42, sbp63)
from projection import sbp_sbp_projection

np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__name__)


# Domain: x = [-1,0], y = [-1,1]
# SBP Subdomain: x = [-1,0], y = [-1,1] (structured mesh)

def main(write_output=True, order=4):
    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    dt_factor = 10
    h = 1/40

    c = np.array([1, 0])
    norm_c = la.norm(c)

    def f(x):
        return sym.sin(5*x)

    def u_analytic(x):
        return f(-(c[0]*x[0] + c[0]*x[1])/norm_c+sym.var("t", sym.DD_SCALAR)*norm_c)

    final_time = 1.0

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    # SBP Half.

    # First, need to set up the structured mesh.
    n_sbp_x = 40
    n_sbp_y = 40
    x_sbp = np.linspace(-np.pi, np.pi, n_sbp_x, endpoint=True)
    y_sbp = np.linspace(-np.pi, np.pi, n_sbp_y, endpoint=True)
    dx = x_sbp[1] - x_sbp[0]
    dy = y_sbp[1] - y_sbp[0]

    # Set up solution vector:
    # For now, timestep is the same as DG.
    u_sbp = parray.zeros(queue, int(n_sbp_x*n_sbp_y), np.float64)

    # Initial condition
    for j in range(0, n_sbp_y):
        for i in range(0, n_sbp_x):
            u_sbp[i + j*(n_sbp_x)] = np.sin(5*(-(c[0]*x_sbp[i]
                                                 + c[0]*y_sbp[j])/norm_c))

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

    # Initial energy
    P_full = np.kron(np.eye(n_sbp_y), dx*p_x).dot(
            np.kron(dy*p_y, np.eye(n_sbp_x)))
    u_work = u_sbp.get()
    u_energy = u_work.dot(P_full.dot(u_work))

    # We are trying to prove if this is the only valid tau-value.
    tau_l = 1
    tau_r = 1/2

    # for the boundaries
    el_x = np.zeros(n_sbp_x)
    er_x = np.zeros(n_sbp_x)
    el_x[0] = 1
    er_x[n_sbp_x-1] = 1
    e_l_matx = np.zeros((n_sbp_x, n_sbp_x,))
    e_r_matx = np.zeros((n_sbp_x, n_sbp_x,))
    e_t_matx = np.zeros((n_sbp_x, n_sbp_x,))
    e_t_matx[0, n_sbp_x-1] = 1.0

    for i in range(0, n_sbp_x):
        for j in range(0, n_sbp_x):
            e_l_matx[i, j] = el_x[i]*el_x[j]
            e_r_matx[i, j] = er_x[i]*er_x[j]

    E_ox = np.kron(np.eye(n_sbp_y), e_l_matx)
    E_nx = np.kron(np.eye(n_sbp_y), e_r_matx)
    E_tx = np.kron(np.eye(n_sbp_y), e_t_matx)
    boldp_y = np.kron(dy*p_y, np.eye(n_sbp_x))

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

    # for the boundaries
    c_l_x = np.kron(tau_l, (np.linalg.inv(dx*p_x).dot(el_x)))

    # For speed...
    dudx_mat = np.kron(np.eye(n_sbp_y), d_x)

    # FIXME: need SBP east to SBP west projection
    east2west, west2east = sbp_sbp_projection(n_sbp_y-1, order_sbp,
                                              n_sbp_y-1, order_sbp)

    def rhs(t, u):
        # Initialize the entire RHS to 0.
        rhs_out = parray.zeros(queue,
                               int(n_sbp_x*n_sbp_y)+1, np.float64)

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
        rhs_sbp = -c[0]*dudx - dl_x
        # Time rate of change of energy estimate from proof.
        # FINAL STEP:
        rhs_energy = u_sbp_ts.dot(((c[0]*E_ox - 2*tau_l*E_ox
                                    + 2*tau_l*E_tx.dot(E_nx) - c[0]*E_nx).dot(
                                        boldp_y)).dot(u_sbp_ts))
        # Now pop this back into the device RHS vector.
        rhs_sbp_dev = parray.zeros(queue, (n_sbp_x*n_sbp_y,), np.float64)
        rhs_sbp_dev = parray.to_device(queue, rhs_sbp)

        rhs_out[0:int(n_sbp_x*n_sbp_y)] = rhs_sbp_dev
        rhs_out[int(n_sbp_x*n_sbp_y)] = rhs_energy

        return rhs_out

    # Timestepper.
    from grudge.shortcuts import set_up_rk4

    # Create timestepper.
    u_comb = parray.zeros(queue, int(n_sbp_x*n_sbp_y)+1, np.float64)
    u_comb[0:int(n_sbp_x*n_sbp_y)] = u_sbp
    u_comb[int(n_sbp_x*n_sbp_y)] = u_energy
    dt_stepper = set_up_rk4("u", dt, u_comb, rhs)

    step = 0

    # Create mesh for structured grid output
    sbp_mesh = np.zeros((2, n_sbp_y, n_sbp_x))
    for j in range(0, n_sbp_y):
        sbp_mesh[0, j, :] = x_sbp
    for i in range(0, n_sbp_x):
        sbp_mesh[1, :, i] = y_sbp

    t_plt = []
    energy_plt = []
    energy_ode_plt = []

    for event in dt_stepper.run(t_end=final_time):
        if isinstance(event, dt_stepper.StateComputed):

            step += 1

            last_t = event.t
            u_sbp = event.state_component[0:int(n_sbp_x*n_sbp_y)]
            u_energy = event.state_component[int(n_sbp_x*n_sbp_y)]

            sbp_error = np.zeros((n_sbp_x*n_sbp_y))
            error_l2_sbp = 0
            for j in range(0, n_sbp_y):
                for i in range(0, n_sbp_x):
                    sbp_error[i + j*n_sbp_x] = u_sbp[i + j*n_sbp_x].get() - \
                        np.sin(5*(-(c[0]*x_sbp[i] + c[0]*y_sbp[j])/norm_c +
                                  last_t*norm_c))
                    error_l2_sbp = error_l2_sbp + \
                        dx*dy*(sbp_error[i + j*n_sbp_x]) ** 2

            error_l2_sbp = np.sqrt(error_l2_sbp)
            print('SBP L2 Error after Step ', step, error_l2_sbp)

            # Calculate energy observed from state)
            u_work = u_sbp.get()
            state_energy = u_work.dot(P_full.dot(u_work))

            # Add exact energy.
            print('Energy from Integrated ODE: ', u_energy)
            print('Energy from State: ', state_energy)

            t_plt.append(last_t)
            energy_plt.append(state_energy)
            energy_ode_plt.append(u_energy.get())

            # Try writing out a VTK file with the SBP data.
            from pyvisfile.vtk import write_structured_grid

            # Overwrite existing files - this is annoying when debugging.
            filename = "Results/advec_proof_%04d.vts" % step
            import os
            if os.path.exists(filename):
                os.remove(filename)

            write_structured_grid(filename, sbp_mesh,
                                  point_data=[("u", u_sbp.get())])

    # Plot the energy data.
    import matplotlib.pyplot as plt

    t_plt_arr = np.array(t_plt)
    e_plt_arr = np.array(energy_plt)
    e_o_plt_arr = np.array(energy_ode_plt)

    plt.plot(t_plt_arr, e_plt_arr, label="State Energy")
    plt.plot(t_plt_arr, e_o_plt_arr, label="ODE Energy")
    plt.legend()
    plt.title("Advection Example - Energy Comparison")
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.savefig("advection_energy.png")
    plt.clf()

    plt.plot(t_plt_arr, abs(e_plt_arr-e_o_plt_arr))
    plt.title("Advection Example - Energy Error")
    plt.xlabel("t")
    plt.ylabel("Energy Error")
    plt.savefig("advection_energy_error.png")


if __name__ == "__main__":
    main()
