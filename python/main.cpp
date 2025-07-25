/* =================================================================================================

(c - GPLv3) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/GooseFEM

================================================================================================= */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include <Eigen/Eigen>

#define GOOSEFEM_USE_XTENSOR_PYTHON
#include "Allocate.hpp"
#include "Element.hpp"
#include "ElementCohesive4.hpp"
#include "ElementHex8.hpp"
#include "ElementQuad4.hpp"
#include "ElementQuad4Axisymmetric.hpp"
#include "ElementQuad4Planar.hpp"
#include "Iterate.hpp"
#include "MatCohesive2d.hpp"
#include "Matrix.hpp"
#include "MatrixDiagonal.hpp"
#include "MatrixDiagonalPartitioned.hpp"
#include "MatrixPartitioned.hpp"
#include "MatrixPartitionedTyings.hpp"
#include "Mesh.hpp"
#include "MeshCohesive.hpp"
#include "MeshCohesiveQuad4.hpp"
#include "MeshHex8.hpp"
#include "MeshQuad4.hpp"
#include "MeshTri3.hpp"
#include "TyingsPeriodic.hpp"
#include "Vector.hpp"
#include "VectorPartitioned.hpp"
#include "VectorPartitionedTyings.hpp"
#include "assertions.hpp"
#include "version.hpp"

namespace py = pybind11;

/**
 * Overrides the `__name__` of a module.
 * Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
 * which affects the `__repr__` of the class type objects.
 */
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

PYBIND11_MODULE(_GooseFEM, m)
{
    // Ensure members to display as `GooseFEM.X` (not `GooseFEM._GooseFEM.X`)
    ScopedModuleNameOverride name_override(m, "GooseFEM");

    xt::import_numpy();

    // --------
    // GooseFEM
    // --------

    m.doc() = "Some simple finite element meshes and operations";

    init_version(m);
    init_assertions(m);
    init_Allocate(m);
    init_Vector(m);
    init_VectorPartitioned(m);
    init_VectorPartitionedTyings(m);
    init_Matrix(m);
    init_MatrixPartitioned(m);
    init_MatrixPartitionedTyings(m);
    init_MatrixDiagonal(m);
    init_MatrixDiagonalPartitioned(m);

    // ----------------
    // GooseFEM.ConstitutiveModels.Cohesive2d
    // ----------------    

    py::module mCohesive = m.def_submodule("ConstitutiveModels", "Constitutive models including cohesive zones");
    
    init_MatCohesive2d(mCohesive);

    // ----------------
    // GooseFEM.Iterate
    // ----------------

    py::module mIterate = m.def_submodule("Iterate", "Iteration support tools");

    init_Iterate(mIterate);

    // ----------------
    // GooseFEM.Element
    // ----------------

    py::module mElement = m.def_submodule("Element", "Generic element routines");

    init_Element(mElement);

    // ----------------------
    // GooseFEM.Element.Cohesive4
    // ----------------------

    py::module mElementCohesive4 =
        mElement.def_submodule("Cohesive4", "Cohesive zone elements (2D)");
    py::module mElementCohesive4Gauss = mElementCohesive4.def_submodule("Gauss", "Gauss quadrature");        

    init_ElementCohesive4(mElementCohesive4);
    init_ElementCohesive4Gauss(mElementCohesive4Gauss);

    // ----------------------
    // GooseFEM.Element.Quad4
    // ----------------------

    py::module mElementQuad4 =
        mElement.def_submodule("Quad4", "Linear quadrilateral elements (2D)");
    py::module mElementQuad4Gauss = mElementQuad4.def_submodule("Gauss", "Gauss quadrature");
    py::module mElementQuad4Nodal = mElementQuad4.def_submodule("Nodal", "Nodal quadrature");
    py::module mElementQuad4MidPoint =
        mElementQuad4.def_submodule("MidPoint", "MidPoint quadrature");

    init_ElementQuad4(mElementQuad4);
    init_ElementQuad4Planar(mElementQuad4);
    init_ElementQuad4Axisymmetric(mElementQuad4);
    init_ElementQuad4Gauss(mElementQuad4Gauss);
    init_ElementQuad4Nodal(mElementQuad4Nodal);
    init_ElementQuad4MidPoint(mElementQuad4MidPoint);

    // ---------------------
    // GooseFEM.Element.Hex8
    // ---------------------

    py::module mElementHex8 =
        mElement.def_submodule("Hex8", "Linear hexahedron (brick) elements (3D)");
    py::module mElementHex8Gauss = mElementHex8.def_submodule("Gauss", "Gauss quadrature");
    py::module mElementHex8Nodal = mElementHex8.def_submodule("Nodal", "Nodal quadrature");

    init_ElementHex8(mElementHex8);
    init_ElementHex8Gauss(mElementHex8Gauss);
    init_ElementHex8Nodal(mElementHex8Nodal);

    // -------------
    // GooseFEM.Mesh
    // -------------

    py::module mMesh = m.def_submodule("Mesh", "Generic mesh routines");

    init_Mesh(mMesh); 

    // ------------------
    // GooseFEM.Mesh.Tri3
    // ------------------

    py::module mMeshTri3 = mMesh.def_submodule("Tri3", "Linear triangular elements (2D)");

    init_MeshTri3(mMeshTri3);

    // -------------------
    // GooseFEM.Mesh.Quad4
    // -------------------

    py::module mMeshQuad4 = mMesh.def_submodule("Quad4", "Linear quadrilateral elements (2D)");

    init_MeshQuad4(mMeshQuad4);

    py::module mMeshQuad4Map = mMeshQuad4.def_submodule("Map", "Map mesh objects");

    init_MeshQuad4Map(mMeshQuad4Map);

    // ------------------
    // GooseFEM.Mesh.Hex8
    // ------------------

    py::module mMeshHex8 = mMesh.def_submodule("Hex8", "Linear hexahedron (brick) elements (3D)");

    init_MeshHex8(mMeshHex8);

    // -------------
    // GooseFEM.CohesiveMesh
    // -------------

    py::module mMeshCohesive = m.def_submodule("MeshCohesive", "Regular cohesive zone mesh (2D)");  

    // -------------
    // GooseFEM.CohesiveMesh.CohesiveQuad4
    // -------------

    py::module mMeshCohesiveQuad4 = mMeshCohesive.def_submodule("Quad4", "Regular cohesive zone mesh (2D)");

    init_MeshCohesiveQuad4(mMeshCohesiveQuad4); 

    // ---------------
    // GooseFEM.Tyings
    // ---------------

    py::module mTyings = m.def_submodule("Tyings", "Linear tying relations");

    init_TyingsPeriodic(mTyings);
}
