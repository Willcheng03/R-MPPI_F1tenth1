#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <cuda_runtime.h>  // cudaStream_t, make_float2
#include <cmath>
#include <array>
#include <map>
#include <string>
#include <vector>

#include <mppi/dynamics/double_integrator/di_dynamics.cuh>
#include <mppi/dynamics/dynamics.cuh>

#include <mppi/cost_functions/quadratic_cost/quadratic_cost.cuh>
#include <mppi/controllers/MPPI/mppi_controller.cuh>
#include <mppi/sampling_distributions/gaussian/gaussian.cuh>

#include <mppi/utils/logger.hpp>

namespace py = pybind11;

// ============================================================
// Null feedback controller (must satisfy Controller interface)
// ============================================================

template <class DYN_T, int N_TIMESTEPS>
class NullFeedback
{
public:
  struct TEMPLATED_PARAMS
  {
  };

  using TEMPLATED_GPU_FEEDBACK   = GPUState;
  using TEMPLATED_FEEDBACK_STATE = GPUState;
  static const int FB_TIMESTEPS  = N_TIMESTEPS;

  using state_array        = typename DYN_T::state_array;
  using control_array      = typename DYN_T::control_array;
  using control_trajectory = Eigen::Matrix<float, DYN_T::CONTROL_DIM, N_TIMESTEPS>;
  using state_trajectory   = Eigen::Matrix<float, DYN_T::STATE_DIM,   N_TIMESTEPS>;

  NullFeedback(DYN_T* /*model*/, float /*dt*/ = 0.01f) {}

  void GPUSetup() {}
  void freeCudaMem() {}
  void initTrackingController() {}

  // Required by Controller::setCUDAStream()
  void bindToStream(cudaStream_t /*stream*/) {}
  // Some code paths may use this name; keep it too.
  void setCUDAStream(cudaStream_t /*stream*/) {}

  void setParams(const TEMPLATED_PARAMS& p) { params_ = p; }
  TEMPLATED_PARAMS getParams() const { return params_; }

  void setDt(float /*dt*/) {}

  void setLogLevel(const mppi::util::LOG_LEVEL& /*level*/) {}
  void setLogger(const mppi::util::MPPILoggerPtr& /*logger*/) {}

  TEMPLATED_FEEDBACK_STATE getFeedbackState() const { return TEMPLATED_FEEDBACK_STATE{}; }

  control_array k(const Eigen::Ref<const state_array>& /*x_act*/,
                  const Eigen::Ref<const state_array>& /*x_goal*/,
                  int /*t*/)
  {
    return control_array::Zero();
  }

  control_array interpolateFeedback_(const Eigen::Ref<const state_array>& /*state*/,
                                    const Eigen::Ref<const state_array>& /*goal_state*/,
                                    double /*rel_time*/,
                                    TEMPLATED_FEEDBACK_STATE& /*fb_state*/)
  {
    return control_array::Zero();
  }

  void computeFeedback(const Eigen::Ref<const state_array>& /*init_state*/,
                       const Eigen::Ref<const state_trajectory>& /*goal_traj*/,
                       const Eigen::Ref<const control_trajectory>& /*control_traj*/)
  {
  }

private:
  TEMPLATED_PARAMS params_{};
};

// ============================================================
// A) Double Integrator MPPI (kept)
// ============================================================

static constexpr int DI_TIMESTEPS    = 65;
static constexpr int DI_NUM_ROLLOUTS = 256;

using DI_DYN     = DoubleIntegratorDynamics;
using DI_COST    = QuadraticCost<DI_DYN>;
using DI_SAMPLER = mppi::sampling_distributions::GaussianDistribution<DI_DYN::DYN_PARAMS_T>;
using DI_FB      = NullFeedback<DI_DYN, DI_TIMESTEPS>;
using DI_CTRL    = VanillaMPPIController<DI_DYN, DI_COST, DI_FB, DI_TIMESTEPS, DI_NUM_ROLLOUTS, DI_SAMPLER>;

class DoubleIntegratorMPPI
{
public:
  DoubleIntegratorMPPI(float dt, float lambda, float alpha, int max_iter)
    : dt_(dt),
      lambda_(lambda),
      alpha_(alpha),
      max_iter_(max_iter),
      model_(),
      cost_(),
      fb_(&model_, dt_),
      sampler_(makeSampler_()),
      controller_(&model_, &cost_, &fb_, &sampler_, dt_, max_iter_, lambda_, alpha_)
  {
    auto params = cost_.getParams();
    for (int i = 0; i < DI_DYN::STATE_DIM; i++)
    {
      params.s_goal[i]   = 0.0f;
      params.s_coeffs[i] = 1.0f;
    }
    cost_.setParams(params);

    auto cparams = controller_.getParams();
    cparams.dynamics_rollout_dim_ = dim3(32, 1, 1);
    cparams.cost_rollout_dim_     = dim3(32, 1, 1);
    controller_.setParams(cparams);
  }

  void set_goal(const Eigen::Vector4f& x_goal, const Eigen::Vector4f& q_coeffs)
  {
    auto params = cost_.getParams();
    for (int i = 0; i < DI_DYN::STATE_DIM; i++)
    {
      params.s_goal[i]   = x_goal[i];
      params.s_coeffs[i] = q_coeffs[i];
    }
    cost_.setParams(params);
  }

  Eigen::Vector2f compute_control(const Eigen::Vector4f& x)
  {
    DI_DYN::state_array s;
    s << x[0], x[1], x[2], x[3];

    controller_.computeControl(s, 1);

    auto u_seq = controller_.getControlSeq();
    Eigen::Vector2f u;
    u[0] = u_seq(0, 0);
    u[1] = u_seq(1, 0);
    return u;
  }

  void slide(int n = 1) { controller_.slideControlSequence(n); }
  float baseline_cost() const { return controller_.getBaselineCost(); }

private:
  static DI_SAMPLER makeSampler_()
  {
    DI_SAMPLER::SAMPLING_PARAMS_T sp;
    for (int i = 0; i < DI_DYN::CONTROL_DIM; i++)
      sp.std_dev[i] = 0.5f;
    return DI_SAMPLER(sp);
  }

  float dt_;
  float lambda_;
  float alpha_;
  int max_iter_;

  DI_DYN model_;
  DI_COST cost_;
  DI_FB fb_;
  DI_SAMPLER sampler_;
  DI_CTRL controller_;
};

// ============================================================
// B) Kinematic Bicycle (F1TENTH) MPPI
//    State:  [x, y, yaw, v]
//    Control: [steer, speed_cmd]
// ============================================================

struct KinematicBicycleParams : DynamicsParams
{
  enum class StateIndex : int { POS_X = 0, POS_Y, YAW, VEL, NUM_STATES };
  enum class ControlIndex : int { STEER = 0, SPEED_CMD, NUM_CONTROLS };
  enum class OutputIndex : int { POS_X = 0, POS_Y, YAW, VEL, NUM_OUTPUTS };

  float wheelbase = 0.33f;
  float tau_speed = 0.25f;
};

class KinematicBicycleDynamics
  : public MPPI_internal::Dynamics<KinematicBicycleDynamics, KinematicBicycleParams>
{
public:
  using BASE          = MPPI_internal::Dynamics<KinematicBicycleDynamics, KinematicBicycleParams>;
  using state_array   = typename BASE::state_array;
  using control_array = typename BASE::control_array;

  KinematicBicycleDynamics(cudaStream_t stream = 0) : BASE(stream)
  {
    std::array<float2, CONTROL_DIM> rng;
    rng[(int)KinematicBicycleParams::ControlIndex::STEER]     = make_float2(-0.418f, 0.418f);
    rng[(int)KinematicBicycleParams::ControlIndex::SPEED_CMD] = make_float2(0.0f, 3.0f);
    this->setControlRanges(rng, false);

    std::array<float, CONTROL_DIM> dead{};
    dead[0] = 0.0f; dead[1] = 0.0f;
    this->setControlDeadbands(dead, false);
  }

  std::string getDynamicsModelName() const override { return "KinematicBicycleDynamics"; }

  // REQUIRED pure-virtual override
  state_array stateFromMap(const std::map<std::string, float>& m) override
  {
    auto pick = [&](const std::vector<std::string>& keys, float fallback) -> float {
      for (const auto& k : keys)
      {
        auto it = m.find(k);
        if (it != m.end()) return it->second;
      }
      return fallback;
    };

    state_array s = state_array::Zero();
    s[(int)KinematicBicycleParams::StateIndex::POS_X] = pick({"x","pos_x","poses_x"}, 0.0f);
    s[(int)KinematicBicycleParams::StateIndex::POS_Y] = pick({"y","pos_y","poses_y"}, 0.0f);
    s[(int)KinematicBicycleParams::StateIndex::YAW]   = pick({"yaw","theta","poses_theta"}, 0.0f);
    s[(int)KinematicBicycleParams::StateIndex::VEL]   = pick({"v","speed","vel","vx","linear_vel","linear_vels_x"}, 0.0f);
    return s;
  }

  void computeKinematics(const Eigen::Ref<const state_array>& state,
                         Eigen::Ref<state_array> s_der)
  {
    for (int i = 0; i < STATE_DIM; i++) s_der[i] = 0.0f;

    const float yaw = state[(int)KinematicBicycleParams::StateIndex::YAW];
    const float v   = state[(int)KinematicBicycleParams::StateIndex::VEL];

    s_der[(int)KinematicBicycleParams::StateIndex::POS_X] = v * std::cos(yaw);
    s_der[(int)KinematicBicycleParams::StateIndex::POS_Y] = v * std::sin(yaw);
  }

  void computeDynamics(const Eigen::Ref<const state_array>& state,
                       const Eigen::Ref<const control_array>& control,
                       Eigen::Ref<state_array> state_der)
  {
    const float steer = control[(int)KinematicBicycleParams::ControlIndex::STEER];
    const float v_cmd = control[(int)KinematicBicycleParams::ControlIndex::SPEED_CMD];
    const float v     = state[(int)KinematicBicycleParams::StateIndex::VEL];

    const auto p = this->getParams();
    const float L   = (p.wheelbase > 1e-6f) ? p.wheelbase : 1e-6f;
    const float tau = (p.tau_speed > 1e-6f) ? p.tau_speed : 1e-6f;

    state_der[(int)KinematicBicycleParams::StateIndex::YAW] = v * std::tan(steer) / L;
    state_der[(int)KinematicBicycleParams::StateIndex::VEL] = (v_cmd - v) / tau;
  }

  __device__ void computeKinematics(float* state, float* state_der)
  {
    for (int i = 0; i < STATE_DIM; i++) state_der[i] = 0.0f;

    const float yaw = state[(int)KinematicBicycleParams::StateIndex::YAW];
    const float v   = state[(int)KinematicBicycleParams::StateIndex::VEL];

    state_der[(int)KinematicBicycleParams::StateIndex::POS_X] = v * cosf(yaw);
    state_der[(int)KinematicBicycleParams::StateIndex::POS_Y] = v * sinf(yaw);
  }

  __device__ void computeDynamics(float* state, float* control, float* state_der, float* /*theta_s*/ = nullptr)
  {
    const float steer = control[(int)KinematicBicycleParams::ControlIndex::STEER];
    const float v_cmd = control[(int)KinematicBicycleParams::ControlIndex::SPEED_CMD];
    const float v     = state[(int)KinematicBicycleParams::StateIndex::VEL];

    const auto p = this->getParams();
    const float L   = (p.wheelbase > 1e-6f) ? p.wheelbase : 1e-6f;
    const float tau = (p.tau_speed > 1e-6f) ? p.tau_speed : 1e-6f;

    state_der[(int)KinematicBicycleParams::StateIndex::YAW] = v * tanf(steer) / L;
    state_der[(int)KinematicBicycleParams::StateIndex::VEL] = (v_cmd - v) / tau;
  }
};

static constexpr int KB_TIMESTEPS    = 45;
static constexpr int KB_NUM_ROLLOUTS = 2048;

using KB_DYN     = KinematicBicycleDynamics;
using KB_COST    = QuadraticCost<KB_DYN>;
using KB_SAMPLER = mppi::sampling_distributions::GaussianDistribution<KB_DYN::DYN_PARAMS_T>;
using KB_FB      = NullFeedback<KB_DYN, KB_TIMESTEPS>;
using KB_CTRL    = VanillaMPPIController<KB_DYN, KB_COST, KB_FB, KB_TIMESTEPS, KB_NUM_ROLLOUTS, KB_SAMPLER>;

class F1TenthKinematicMPPI
{
public:
  F1TenthKinematicMPPI(float dt,
                       float wheelbase,
                       float tau_speed,
                       float max_speed,
                       float lambda,
                       float alpha,
                       int max_iter)
    : dt_(dt),
      lambda_(lambda),
      alpha_(alpha),
      max_iter_(max_iter),
      model_(),
      cost_(),
      fb_(&model_, dt_),
      sampler_(makeSampler_()),
      controller_(&model_, &cost_, &fb_, &sampler_, dt_, max_iter_, lambda_, alpha_)
  {
    // dynamics params
    auto p = model_.getParams();
    p.wheelbase = wheelbase;
    p.tau_speed = tau_speed;
    model_.setParams(p);

    // control ranges (override speed max)
    std::array<float2, KB_DYN::CONTROL_DIM> rng;
    rng[(int)KinematicBicycleParams::ControlIndex::STEER]     = make_float2(-0.418f, 0.418f);
    rng[(int)KinematicBicycleParams::ControlIndex::SPEED_CMD] = make_float2(0.0f, max_speed);
    model_.setControlRanges(rng, false);

    // default goal + weights
    set_goal(Eigen::Vector4f::Zero(), Eigen::Vector4f(20.f, 20.f, 8.f, 2.f));

    auto cparams = controller_.getParams();
    cparams.dynamics_rollout_dim_ = dim3(32, 1, 1);
    cparams.cost_rollout_dim_     = dim3(32, 1, 1);
    controller_.setParams(cparams);
  }

  void set_goal(const Eigen::Vector4f& x_goal, const Eigen::Vector4f& q_coeffs)
  {
    auto params = cost_.getParams();
    for (int i = 0; i < KB_DYN::STATE_DIM; i++)
    {
      params.s_goal[i]   = x_goal[i];
      params.s_coeffs[i] = q_coeffs[i];
    }
    cost_.setParams(params);
  }

  // input: [x,y,yaw,v] -> output: [steer, speed_cmd]
  Eigen::Vector2f compute_control(const Eigen::Vector4f& x)
  {
    KB_DYN::state_array s;
    s << x[0], x[1], x[2], x[3];

    controller_.computeControl(s, 1);

    auto u_seq = controller_.getControlSeq();
    Eigen::Vector2f u;
    u[0] = u_seq(0, 0);
    u[1] = u_seq(1, 0);
    return u;
  }

  void slide(int n = 1) { controller_.slideControlSequence(n); }
  float baseline_cost() const { return controller_.getBaselineCost(); }

private:
  static KB_SAMPLER makeSampler_()
  {
    KB_SAMPLER::SAMPLING_PARAMS_T sp;
    sp.std_dev[(int)KinematicBicycleParams::ControlIndex::STEER]     = 0.12f;
    sp.std_dev[(int)KinematicBicycleParams::ControlIndex::SPEED_CMD] = 0.80f;
    return KB_SAMPLER(sp);
  }

  float dt_;
  float lambda_;
  float alpha_;
  int max_iter_;

  KB_DYN model_;
  KB_COST cost_;
  KB_FB fb_;
  KB_SAMPLER sampler_;
  KB_CTRL controller_;
};

// ============================================================
// PYBIND module
// ============================================================

PYBIND11_MODULE(rmppi_cuda, m)
{
  m.doc() = "CUDA MPPI bindings (Double Integrator + Kinematic Bicycle for F1TENTH)";

  py::class_<DoubleIntegratorMPPI>(m, "DoubleIntegratorMPPI")
    .def(py::init<float, float, float, int>(),
         py::arg("dt")=0.015f,
         py::arg("lambda")=1.0f,
         py::arg("alpha")=1.0f,
         py::arg("max_iter")=1)
    .def("set_goal", &DoubleIntegratorMPPI::set_goal, py::arg("x_goal"), py::arg("q_coeffs"))
    .def("compute_control", &DoubleIntegratorMPPI::compute_control, py::arg("x"))
    .def("slide", &DoubleIntegratorMPPI::slide, py::arg("n")=1)
    .def("baseline_cost", &DoubleIntegratorMPPI::baseline_cost);

  py::class_<F1TenthKinematicMPPI>(m, "F1TenthKinematicMPPI")
    .def(py::init<float, float, float, float, float, float, int>(),
         py::arg("dt")=0.02f,
         py::arg("wheelbase")=0.33f,
         py::arg("tau_speed")=0.25f,
         py::arg("max_speed")=3.0f,
         py::arg("lambda")=1.0f,
         py::arg("alpha")=1.0f,
         py::arg("max_iter")=1)
    .def("set_goal", &F1TenthKinematicMPPI::set_goal, py::arg("x_goal"), py::arg("q_coeffs"))
    .def("compute_control", &F1TenthKinematicMPPI::compute_control, py::arg("x"))
    .def("slide", &F1TenthKinematicMPPI::slide, py::arg("n")=1)
    .def("baseline_cost", &F1TenthKinematicMPPI::baseline_cost);
}


