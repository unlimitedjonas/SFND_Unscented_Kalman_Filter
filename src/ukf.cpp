#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // the timestamp signal
  time_us_ = 0.0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::Init(MeasurementPackage meas_package){

  // augmented state dimension
  n_aug_ = n_x_ + 2;
  lambda_ = 3 -  n_aug_;

  // Initialize weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5/(lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);     

  // create example matrix with predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
              
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  x_.fill(0.0);

  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    float rho = meas_package.raw_measurements_(0);
    float phi = meas_package.raw_measurements_(1);
    float rho_dot = meas_package.raw_measurements_(2);
    x_(0) = rho * cos(phi);
    x_(1) = rho * sin(phi);

    float vx = rho_dot * cos(phi);
    float vy = rho_dot * sin(phi);
    float v = sqrt(vx*vx + vy*vy);
    x_(3) = v;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
  }
  time_us_ = meas_package.timestamp_;
  is_initialized_ = true;
  return;
}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if(!is_initialized_){
    Init(meas_package);
  }
  
  double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    UpdateLidar(meas_package);
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    UpdateRadar(meas_package);
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  /**
   * augmented sigma points
   */
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0); // P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
      Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
      Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }


  /*
   * predict the sigma points
   */
  for (int i = 0; i < Xsig_aug.cols(); i++) {
    VectorXd pred = Xsig_aug.col(i).head(n_x_);
    VectorXd temp1(n_x_);
    VectorXd temp2(n_x_);

    double velocity = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawrate = Xsig_aug(4, i);
    double mu_a = Xsig_aug(5, i);
    double mu_yawr = Xsig_aug(6, i);

    if (yawrate != 0) {
      temp1 << velocity / yawrate * (sin(yaw + yawrate * delta_t) - sin(yaw)),
        velocity / yawrate * (-cos(yaw + yawrate * delta_t) + cos(yaw)),
        0,
        yawrate * delta_t,
        0;
    }
    // avoid division by zero
    else {
      temp1 << velocity * cos(yaw) * delta_t,
        velocity * sin(yaw) * delta_t,
        0,
        yawrate * delta_t,
        0;
    }

    temp2 << 0.5 * pow(delta_t, 2) * cos(yaw) * mu_a,
      0.5 * pow(delta_t, 2) * sin(yaw) * mu_a,// the 1/2 expression is banned and cause error
      delta_t * mu_a,
      0.5 * pow(delta_t, 2) * mu_yawr,//it should be expressed by 0.5
      delta_t* mu_yawr;

    // write predicted sigma points into right column
    pred = pred + temp1 + temp2;
    Xsig_pred_.col(i) = pred;
  }

  /*
    * predict the mean and covariance
    */
  int n_a = 2 * n_aug_ + 1;   

  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < n_a; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_a; i++) {
    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angel normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

   // the dimension of lidar measurement
  int n_z_ = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // transform sigma points into measurement space
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  // calculate mean predicted measurement 
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R(n_z_, n_z_);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_* std_laspy_;
  S += R;

  // create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // the dimension of radar measurement
  int n_z_ = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    // transform sigma points into measurement space
    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px * cos(yaw) * v + py * sin(yaw) * v) / Zsig(0,i);
  }

  // calculate mean predicted measurement
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization, it is so sophisticated!!!
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R(n_z_, n_z_);
  R << std_radr_ * std_radr_, 0, 0,
    0, std_radphi_* std_radphi_, 0,
    0, 0, std_radrd_* std_radrd_;
  S += R;

  // create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  // calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  //angel normalization
  while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

