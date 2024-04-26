#include "pybind_controller.h"

MJCController::MJCController()
{
	_k = 7;
	Initialize();
}

MJCController::~MJCController()
{
}

void MJCController::read(double t, std::array<double, 9> q, std::array<double, 9> qdot)
{	
	_t = t;
	if (_bool_init == true)
	{
		_init_t = _t;
		_bool_init = false;
	}

	_dt = t - _pre_t;
	_pre_t = t;

	for (int i = 0; i < _k; i++)
	{
		_q(i) = q[i];
		_qdot(i) = qdot[i];
		// _qdot(i) = CustomMath::VelLowpassFilter(0.001, 2.0*PI* 10.0, _pre_q(i), _q(i), _pre_qdot(i)); //low-pass filter
		_pre_q(i) = _q(i);
		_pre_qdot(i) = _qdot(i);
	}

	// to use gripper, need to write some code!

	// OK!
	// cout << "read qpos: \n" << _q << endl;
	// cout<<"read t: " << _t << endl;
	// cout<<"read pre_t: " << _pre_t << endl;
	// cout<<"read dt: " << _dt << endl;
}

std::array<double, 9> MJCController::write()
{
	for (int i = 0; i < _k; i++)
	{
		torque[i] = _torque(i);
		// cout << torque[i] << endl;
	}

	for (int i = 0; i < 1; i++)
	{
		torque[_k + i] = 0.2; // change the value
	}
	// cout <<torque<< "\n==============\n" << endl;
	return torque;
}

void MJCController::control_mujoco(std::array<double, 3> des_position)
{
	// Only Operational Space Control

    ModelUpdate();
    // motionPlan();

	// cout << "present angle:\n" << _q << "\n================" << endl;
	// cout << "present pose:\n" << _x_hand << "\n================" << endl;

	VectorXd target_pose;
	target_pose.setZero(6);
	// target_pose(0) = _x_hand(0) - 0.1;
	// target_pose(1) = _x_hand(1) + 0.05;
	// target_pose(2) = _x_hand(2) + 0.05;
	target_pose(0) = des_position[0];
	target_pose(1) = des_position[1];
	target_pose(2) = des_position[2];
	target_pose(3) = 3.14;
	target_pose(4) = 0;
	target_pose(5) = -0.78;
	// cout << "target pose:\n" << target_pose << "\n================" << endl;

	reset_target(1.0, target_pose);

	// Operational Control
	if (_t - _init_t < 0.1 && _bool_ee_motion == false)
	{
		_start_time = _init_t;
		_end_time = _start_time + _motion_time;
		HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
		HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
		_bool_ee_motion = true;
		// cout<<"_t : "<<_t<<endl;
		// cout<<"ctrl_mujoco/_x_goal_hand :\n"<<_x_goal_hand<<endl;
		// cout<<"ctrl_mujoco/start_time: "<<_start_time<<endl;
		// cout<<"ctrl_mujoco/end_time: "<<_end_time<<endl;
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

	// if(_control_mode == 1) // joint space control
	// {
	// 	if (_t - _init_t < 0.1 && _bool_joint_motion == false)
	// 	{
	// 		_start_time = _init_t;
	// 		_end_time = _start_time + _motion_time;
	// 		JointTrajectory.reset_initial(_start_time, _q, _qdot);
	// 		JointTrajectory.update_goal(_q_goal, _qdot_goal, _end_time);
	// 		_bool_joint_motion = true;

	// 		// cout<<"ctrl_mujoco/_q_goal :\n"<<_q_goal<<endl;
	// 		// cout<<"ctrl_mujoco/start_time: "<<_start_time<<endl;
	// 		// cout<<"ctrl_mujoco/end_time: "<<_end_time<<endl;
	// 	}
		
	// 	JointTrajectory.update_time(_t);
	// 	_q_des = JointTrajectory.position_cubicSpline();
	// 	_qdot_des = JointTrajectory.velocity_cubicSpline();

	// 	JointControl();

	// 	if (JointTrajectory.check_trajectory_complete() == 1)
	// 	{
	// 		_bool_plan(_cnt_plan) = 1;
	// 		_bool_init = true;
	// 	}
	// }

	// else if(_control_mode == 3) // operational space control
	// {
	// 	if (_t - _init_t < 0.1 && _bool_ee_motion == false)
	// 	{
	// 		_start_time = _init_t;
	// 		_end_time = _start_time + _motion_time;
	// 		HandTrajectory.reset_initial(_start_time, _x_hand, _xdot_hand);
	// 		HandTrajectory.update_goal(_x_goal_hand, _xdot_goal_hand, _end_time);
	// 		_bool_ee_motion = true;
	// 		// cout<<"_t : "<<_t<<endl;
	// 		// cout<<"ctrl_mujoco/_x_goal_hand :\n"<<_x_goal_hand<<endl;
	// 		// cout<<"ctrl_mujoco/start_time: "<<_start_time<<endl;
	// 		// cout<<"ctrl_mujoco/end_time: "<<_end_time<<endl;
	// 	}

	// 	HandTrajectory.update_time(_t);
	// 	_x_des_hand.head(3) = HandTrajectory.position_cubicSpline();
	// 	_R_des_hand = HandTrajectory.rotationCubic();
	// 	_x_des_hand.segment<3>(3) = CustomMath::GetBodyRotationAngle(_R_des_hand);
	// 	_xdot_des_hand.head(3) = HandTrajectory.velocity_cubicSpline();
	// 	_xdot_des_hand.segment<3>(3) = HandTrajectory.rotationCubicDot();		

	// 	OperationalSpaceControl();

	// 	if (HandTrajectory.check_trajectory_complete() == 1)
	// 	{
	// 		_bool_plan(_cnt_plan) = 1;
	// 		_bool_init = true;
	// 	}
	// }
	
}

void MJCController::reset_target(double motion_time, VectorXd target_pose)
{
	// operational space control
	_control_mode = 3;
	_motion_time = motion_time;
	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_x_goal_hand = target_pose;
	_xdot_goal_hand.setZero();
}

bool MJCController::check_joint_limit(std::array<double, 9> q)
{
	bool limit = false;

	for(int i = 0; i < _k; i++)
	{
		if(((0.1 + _min_joint_position[i]) < q[i]) && (q[i] < (_max_joint_position[i] - 0.1)))
		{

		}
		else
		{
			limit = true;
		}
	}

	return limit;
}

bool MJCController::check_velocity_limit()
{
	bool limit = false;

	for(int i = 0; i < 3; i++)
	{
		if(abs(_xdot_hand[i]) > 0.8)
		{
			limit = true;
		}
	}
	
	return limit;
}

void MJCController::ModelUpdate()
{
    Model.update_kinematics(_q, _qdot);
	Model.update_dynamics();
    Model.calculate_EE_Jacobians();
	Model.calculate_EE_positions_orientations();
	Model.calculate_EE_velocity();

	// cout << "\033[33mEE Velocity: \n" << Model._xdot_hand << "\033[0m\n" << "=======================" << endl;

	_J_hands = Model._J_hand;

	_x_hand.head(3) = Model._x_hand;
	_x_hand.tail(3) = CustomMath::GetBodyRotationAngle(Model._R_hand);
	_xdot_hand = Model._xdot_hand;
}	

void MJCController::motionPlan()
{	
	if (_bool_plan(_cnt_plan) == 1)
	{
		if(_cnt_plan == 0)
		{	
			cout << "plan: " << _cnt_plan << endl;
			_q_order(0) = _q_home(0);
			_q_order(1) = _q_home(1);
			_q_order(2) = _q_home(2);
			_q_order(3) = _q_home(3);
			_q_order(4) = _q_home(4);
			_q_order(5) = _q_home(5);
			_q_order(6) = _q_home(6);		                    
			// reset_target(10.0, _q_order, _qdot);

			_control_mode = 1;
			_motion_time = 10.0;
			_bool_joint_motion = false;
			_bool_ee_motion = false;

			_q_goal = _q_order.head(7);
			_qdot_goal.setZero();

			_cnt_plan++;
		}

		else if(_cnt_plan == 1)
		{
			cout << "plan: " << _cnt_plan << endl;
			cout << "present angle:\n" << _q << "\n================" << endl;
			cout << "present pose:\n" << _x_hand << "\n================" << endl;

			VectorXd target_pose;
			target_pose.setZero(6);
			// target_pose(0) = _x_hand(0) - 0.1;
			// target_pose(1) = _x_hand(1) + 0.05;
			// target_pose(2) = _x_hand(2) + 0.05;
			target_pose(0) = _x_hand(0) + 0.1;
			target_pose(1) = _x_hand(1) + 0.1;
			target_pose(2) = _x_hand(2) + 0.1;
			target_pose(3) = _x_hand(3);
			target_pose(4) = _x_hand(4);
			target_pose(5) = _x_hand(5);
			cout << "target pose:\n" << target_pose << "\n================" << endl;

 			reset_target(5.0, target_pose);
			_cnt_plan++;
		}

		// else if(_cnt_plan == 2)
		// {
		// 	cout << "plan: " << _cnt_plan << endl;
		// 	cout << "present pose:\n" << _x_hand << "\n================" << endl;

		// 	VectorXd target_pose;
		// 	target_pose.setZero(6);
		// 	// target_pose(0) = _x_hand(0) - 0.1;
		// 	// target_pose(1) = _x_hand(1) + 0.05;
		// 	// target_pose(2) = _x_hand(2) + 0.05;
		// 	target_pose(0) = _x_hand(0) + 0.1;
		// 	target_pose(1) = _x_hand(1) + 0.1;
		// 	target_pose(2) = _x_hand(2) + 0.1;
		// 	target_pose(3) = _x_hand(3);
		// 	target_pose(4) = _x_hand(4);
		// 	target_pose(5) = _x_hand(5);
		// 	cout << "target pose:\n" << target_pose << "\n================" << endl;

 		// 	reset_target(5.0, target_pose);
		// 	_cnt_plan++;
		// }
	}
}

void MJCController::JointControl()
{	
	// _control_mode = 1
	_torque.setZero();
	_A_diagonal = Model._A;
	for(int i = 0; i < 7; i++){
		_A_diagonal(i,i) += 1.0;
	}
	// Manipulator equations of motion in joint space
	_torque = _A_diagonal*(400*(_q_des - _q) + 40*(_qdot_des - _qdot)) + Model._bg;
	// cout<<"_q_des 	 : "<<_q_des.transpose()<<endl;
	// cout<<"_q 		 : "<<_q.transpose()<<endl;
	// cout<<"_qdot_des : "<<_qdot_des.transpose()<<endl;
	// cout<<"_qdot	 : "<<_qdot.transpose()<<endl<<endl;

	// cout << "cmd_torque: \n" << _torque << endl;
}

void MJCController::OperationalSpaceControl()
{
	// _control_mode = 3
	_torque.setZero();

	// calc position, velocity errors
	_x_err_hand.segment(0,3) = _x_des_hand.head(3) - _x_hand.head(3);
	_x_err_hand.segment(3,3) = -CustomMath::getPhi(Model._R_hand, _R_des_hand);
	_x_dot_err_hand.segment(0,3) = _xdot_des_hand.head(3) - _xdot_hand.head(3);
	_x_dot_err_hand.segment(3,3) = _xdot_des_hand.tail(3) - _xdot_hand.tail(3);

	// jacobian pseudo inverse matrix
	_J_bar_hands = CustomMath::pseudoInverseQR(_J_hands);

	// jacobian transpose matrix
	_J_T_hands = _J_hands.transpose();

	// jacobian inverse transpose matrix
	_J_bar_T_hands = CustomMath::pseudoInverseQR(_J_T_hands);

	// Should consider [Null space] cuz franka_panda robot = 7 DOF
	_J_null = _I - _J_T_hands * _J_bar_T_hands;

	// Inertial matrix: operational space
	_Lambda = _J_bar_T_hands * Model._A * _J_bar_hands;

	F_command_star = 400 * _x_err_hand + 40 * _x_dot_err_hand;

	_torque = (_J_T_hands * _Lambda * F_command_star + Model._bg) + _J_null * Model._A * (_qdot_des-_qdot);

	// cout << "cmd_torque: \n" << _torque << endl;

	// cout << "_J_null\n" << _J_null * Model._A << endl;

	// cout << "\ntarget pose:\n" << " x -> x - 0.1\n y -> y + 0.05\n z -> z + 0.05\n===============\n" << endl;
	// cout << "Robot pose error:\n" << _x_des_hand.head(3) - Model._x_hand << "\n"
	// << _x_des_hand.tail(3) - CustomMath::GetBodyRotationAngle(Model._R_hand) << "\n===============\n===============" << endl;
}

void MJCController::Initialize()
{
    _control_mode = 1; //1: joint space, 2: task space(CLIK), 3: operational space

	_bool_init = true;
	_t = 0.0;
	_init_t = 0.0;
	_pre_t = 0.0;
	_dt = 0.0;

	_kpj = 400.0;
	_kdj = 20.0;

	// _kpj_diagonal.setZero(_k, _k);
	// //							0 		1	2		3	   4	5 	6
	// _kpj_diagonal.diagonal() << 400., 2500., 1500., 1700., 700., 500., 520.;
	// _kdj_diagonal.setZero(_k, _k);
	// _kdj_diagonal.diagonal() << 20., 250., 170., 320., 70., 50., 15.;
	_x_kp = 1;//작게 0.1
	// _x_kp = 20.0;
	_x_kd = 1;

    _q.setZero(_k); // use
	_qdot.setZero(_k); // use
	_torque.setZero(_k); // use
	torque = {0}; // use

	_J_hands.setZero(6,_k);
	_J_bar_hands.setZero(_k,6);
	_J_T_hands.setZero(_k, 6);

	_x_hand.setZero(6);
	_xdot_hand.setZero(6);

	//////////////////원본///////////////////
	// _cnt_plan = 0;
	_bool_plan.setZero(30);
	// _time_plan.resize(30);
	// _time_plan.setConstant(5.0);
	//////////////////원본///////////////////

	_q_home.setZero(_k);
	// _q_home(0) = 0.0;
	// _q_home(1) = -30.0 * DEG2RAD;
	// _q_home(2) = 30.0 * DEG2RAD;
	// _q_home(3) = -30.0 * DEG2RAD;
	// _q_home(4) = 30.0 * DEG2RAD;
	// _q_home(5) = -60.0 * DEG2RAD;
	// _q_home(6) = 30.0 * DEG2RAD;
	_q_home(0) = 0;
	_q_home(1) = -M_PI_4;
	_q_home(2) = 0;
	_q_home(3) = -3 * M_PI_4;
	_q_home(4) = 0;
	_q_home(5) = M_PI_2;
	_q_home(6) = M_PI_4;

	_start_time = 0.0;
	_end_time = 0.0;
	_motion_time = 0.0;

	_bool_joint_motion = false;
	_bool_ee_motion = false;

	_q_des.setZero(_k);
	_qdot_des.setZero(_k);
	_q_goal.setZero(_k);
	_qdot_goal.setZero(_k);

	_x_des_hand.setZero(6);
	_xdot_des_hand.setZero(6);
	_x_goal_hand.setZero(6);
	_xdot_goal_hand.setZero(6);

	_pos_goal_hand.setZero(); // 3x1 
	_rpy_goal_hand.setZero(); // 3x1
	JointTrajectory.set_size(_k);
	_A_diagonal.setZero(_k,_k);

	_x_err_hand.setZero(6);
	_x_dot_err_hand.setZero(6);
	_R_des_hand.setZero();

	_I.setIdentity(7,7);
	_J_null.setZero(_k,_k);

	_pre_q.setZero(7); // use
	_pre_qdot.setZero(7); // use

	///////////////////save_stack/////////////////////
	_q_order.setZero(7);
	_qdot_order.setZero(7);
	// _max_joint_position.setZero(7);
	// _min_joint_position.setZero(7);

	// _min_joint_position(0) = -2.9671;
	// _min_joint_position(1) = -1.8326;
	// _min_joint_position(2) = -2.9671;
	// _min_joint_position(3) = -3.1416;
	// _min_joint_position(4) = -2.9671;
	// _min_joint_position(5) = -0.0873;
	// _min_joint_position(6) = -2.9671;

	// _max_joint_position(0) = 2.9671;
	// _max_joint_position(1) = 1.8326;
	// _max_joint_position(2) = 2.9671;
	// _max_joint_position(3) = 0.0;
	// _max_joint_position(4) = 2.9671;
	// _max_joint_position(5) = 3.8223;
	// _max_joint_position(6) = 2.9671;

	///////////////////estimate_lr/////////////////////

	// cout << fixed;
	// cout.precision(3);

	// For joint limit
	_max_joint_position.setZero(7);
	_min_joint_position.setZero(7);
	_max_joint_position = Model._max_joint_position;
	_min_joint_position = Model._min_joint_position;

	_cnt_plan = 0;
	_bool_plan(_cnt_plan) = 1;
}

namespace py = pybind11;
PYBIND11_MODULE(mjc_controller, m)
{
	m.doc() = "pybind11 for controller";

	py::class_<MJCController>(m, "MJCController")
		.def(py::init<>())
		.def("read", &MJCController::read)
		.def("control_mujoco", &MJCController::control_mujoco)
		.def("write", &MJCController::write)
		.def("joint_limit", &MJCController::check_joint_limit)
		.def("velocity_limit", &MJCController::check_velocity_limit)
		;

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}

// int add(int i, int j) {  return i + j; }
// int sub(int i, int j) {  return i - j; }

// struct MyData
// {
//   float x, y;

//   MyData() : x(0), y(0) { }
//   MyData(float x, float y) : x(x), y(y) { }

//   void print() { printf("%f, %f\n", x, y); }
// };

// PYBIND11_MODULE(mjc_controller, m) {          // "example" module name should be same of module name in CMakeLists
//   m.doc() = "pybind11 example plugin"; // optional module docstring

//   m.def("add", &add, "A function that adds two numbers");
//   m.def("sub", &sub, "A function that subtracts two numbers");

//   py::class_<MyData>(m, "MyData")
//     .def(py::init<>())
//     .def(py::init<float, float>(), "constructor 2", py::arg("x"), py::arg("y"))
//     .def("print", &MyData::print)
//     .def_readwrite("x", &MyData::x)
//     .def_readwrite("y", &MyData::y);
// }



// cout<< "goal_pos: \n" << _goal_pos << endl;
// cout<< "goal_vel: \n" << _goal_vel << endl;
// cout<< "goal_time: \n" << goal_time << endl;