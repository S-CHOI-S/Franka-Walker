#include "pybind_controller.h"

CController::CController()
{
	_k = 9;
	Initialize();
}

CController::~CController()
{
}

void CController::read(double t, std::array<double, 9> q, std::array<double, 9> qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	// cout<<"_dt : "<<_dt<<endl;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];
		// _qdot(i) = CustomMath::VelLowpassFilter(0.001, 2.0*PI* 10.0, _pre_q(i), _q(i), _pre_qdot(i)); //low-pass filter
		_pre_q(i) = _q(i);
		_pre_qdot(i) = _qdot(i);	
			
		// if(_t < 2.0)///use filtered data after convergece
        // {
		// 	_qdot(i) = qdot[i];
		// }
	}
}

void CController::write(std::array<double, 9> torque)
{
	for (int i = 0; i < _k; i++)
	{
		torque[i] = _torque(i);
	}
}

void CController::control_mujoco()
{
    ModelUpdate();
    motionPlan();
	if(_control_mode == 1) // joint space control
	{
		if (_t - _init_t < 0.1 && _bool_joint_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			JointTrajectory.reset_initial(_start_time, _q, _qdot);
			JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
			_bool_joint_motion = true;
		}
		
		JointTrajectory.update_time(_t);
		_q_des = JointTrajectory.position_cubicSpline();
		_qdot_des = JointTrajectory.velocity_cubicSpline();

		JointControl();

		if (JointTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if(_control_mode == 2) // inverse kinematics control (CLIK)
	{		
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;
			// cout<<"_t : "<<_t<<endl;
			// cout<<"_x_hand 	: "<<_x_hand.transpose()<<endl;
		}

		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

		CLIK();

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
	else if(_control_mode == 3) // operational space control
	{
		if (_t - _init_t < 0.1 && _bool_ee_motion == false)
		{
			_start_time = _init_t;
			_end_time = _start_time + _motion_time;
			HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
			HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
			_bool_ee_motion = true;
			// cout<<"_t : "<<_t<<endl;
			// cout<<"_x_hand 	: "<<_x_hand.transpose()<<endl;
		}

		HandTrajectory.update_time(_t);
		_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
		_R_des_hand = HandTrajectory.rotationCubic();
		_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
		_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
		_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

		OperationalSpaceControl();

		if (HandTrajectory.check_trajectory_complete() == 1)
		{
			_bool_plan(_cnt_plan) = 1;
			_bool_init = true;
		}
	}
}

namespace py = pybind11;
// PYBIND11_MODULE(controller, m)
// {
// 	m.doc() = "pybind11 for controller";

// 	py::class_<CController>(m, "CController")
// 		.def(py::init<>())
// 		.def("read", &CController::read)
// 		.def("control_mujoco", &CController::control_mujoco)
// 		.def("write", &CController::write);

// #ifdef VERSION_INFO
// 	m.attr("__version__") = VERSION_INFO;
// #else
// 	m.attr("__version__") = "dev";
// #endif
// }

int add(int i, int j) {  return i + j; }
int sub(int i, int j) {  return i - j; }

struct MyData
{
  float x, y;

  MyData() : x(0), y(0) { }
  MyData(float x, float y) : x(x), y(y) { }

  void print() { printf("%f, %f\n", x, y); }
};

PYBIND11_MODULE(mjc_controller, m) {          // "example" module name should be same of module name in CMakeLists
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function that adds two numbers");
  m.def("sub", &sub, "A function that subtracts two numbers");

  py::class_<MyData>(m, "MyData")
    .def(py::init<>())
    .def(py::init<float, float>(), "constructor 2", py::arg("x"), py::arg("y"))
    .def("print", &MyData::print)
    .def_readwrite("x", &MyData::x)
    .def_readwrite("y", &MyData::y);
}