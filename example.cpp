#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <robotmodel.h>

class A
{
    public:
        A();
        ~A();

        int add(int i, int j);

    private:
        

};

A::A()
{

}

A::~A()
{

}

int A::add(int i, int j)
{
    return i + j;
}

class B
{
    public:
        B();
        ~B();

        int here(int i, int j);
        int acalc(int i, int j);
        // CModel Model;

    private:
        A a;
        CModel Model;
        
};

B::B()
{

}

B::~B()
{

}

int B::here(int i, int j) 
{
    return i + j;
}

int B::acalc(int i, int j)
{
    std::cout<<"Class B::acalc!"<<endl;
    return a.add(i,j);
}


namespace py = pybind11;
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<B>(m, "B")
        .def(py::init<>())
        .def("here", &B::here, "A function that adds two numbers")
        .def("acalc", &B::acalc, "A function that adds two numbers")
        // .def_readwrite("Model", &B::Model)
        ;
}
