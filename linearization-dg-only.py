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
import numpy.linalg as la

from grudge.array_context import PyOpenCLArrayContext

from arraycontext.container.traversal import thaw
import meshmode.mesh.generation as mgen

from grudge import DiscretizationCollection

import grudge.dof_desc as dof_desc

import pyopencl as cl
import pyopencl.tools as cl_tools

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
    actx = PyOpenCLArrayContext(
        queue,
        allocator=cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue)),
        force_device_scalars=True,
    )

    # DG Half.
    dim = 2

    nelem_x = 10
    nelem_y = 10
    mesh = mgen.generate_regular_rect_mesh(a=(0, -1), b=(1, 1),
                                           n=(nelem_x, nelem_y),
                                           order=order,
                                           boundary_tag_to_face={
                                               "btag_std": ["+y", "-x",
                                                            "+x", "-y"]})

    dcoll = DiscretizationCollection(actx, mesh, order=order)
    dt_factor = 4
    h = 1/40

    c = np.array([0.1, 0.1])
    norm_c = la.norm(c)

    flux_type = "upwind"

    def f(x):
        return actx.np.sin(10*x)

    def u_analytic(x, t=0):
        return f(-c.dot(x)/norm_c+t*norm_c)

    from grudge.models.advection import WeakAdvectionOperator
    std_tag = dof_desc.DTAG_BOUNDARY("btag_std")

    adv_operator = WeakAdvectionOperator(dcoll, c,
                                         inflow_u=lambda t: u_analytic(
                                             thaw(dcoll.nodes(
                                                 dd=std_tag),
                                                 actx),
                                             t=t
                                             ),
                                         flux_type=flux_type)

    nodes = thaw(dcoll.nodes(), actx)
    u = u_analytic(nodes, t=0)

    # Count the number of DG nodes - will need this later
    ngroups = len(mesh.groups)
    nnodes_grp = np.ones(ngroups)
    for i in range(0, dim):
        for j in range(0, ngroups):
            nnodes_grp[j] = nnodes_grp[j] * \
                    mesh.groups[j].nodes.shape[i*-1 - 1]

    nnodes = int(sum(nnodes_grp))

    final_time = 0.2

    dt = dt_factor * h/order**2
    nsteps = (final_time // dt) + 1
    dt = final_time/nsteps + 1e-15

    def rhs(t, u):
        return adv_operator.operator(t, u)

    # Now, instead of timestepping, we'll make
    # ourselves a massive RHS operator,
    # and (hopefully) find its eigenvalues.

    u_base = u

    # Get base RHS
    base_rhs = rhs(0, u_base)

    # Perturbation.
    eps = 1e-9

    # Our beautiful matrix.
    rhs_op_dense = np.zeros((nnodes, nnodes))
    rhs_op = sparse.csc_matrix(rhs_op_dense)
    rhs_op = rhs_op.tolil()
    print('Initialized rhs operator')
    print('Dimension: ', nnodes)
    print('Entering point loop')
    from meshmode.dof_array import flatten, unflatten

    for i in range(0, nnodes):

        print('Calculating RHS for perturbed point ', i)
        u_pert = u_base.copy()
        u_pert_mod = flatten(u_pert)
        u_pert_mod[i] += eps
        u_pert = unflatten(actx, dcoll.discr_from_dd("vol"),
                           actx.from_numpy(u_pert_mod.get()))
        pert_rhs = rhs(0, u_pert)

        # RHS diff...
        pert_vec = (flatten(pert_rhs) - flatten(base_rhs)) / eps

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
    sparse.save_npz("{npoints}x{npoints}_dg_only.npz".format(npoints=nnodes),
                    rhs_op_csc)


if __name__ == "__main__":
    main()
