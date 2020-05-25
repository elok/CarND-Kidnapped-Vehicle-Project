/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;

// declare a random engine to be used across multiple and various method calls
std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  // Standard deviations for x, y, and theta
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2]; 

  // Creates normal (Gaussian) distributions for x, y, and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for(int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Creates normal (Gaussian) distributions for ...
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for(int i = 0; i<num_particles; i++){

    if (fabs(yaw_rate) < 0.00001) {
      // Yaw has not changed
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else {
      // Yaw has changed
      particles[i].x += velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate * delta_t ) - sin(particles[i].theta) );
      particles[i].y += velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t) );
      particles[i].theta += yaw_rate * delta_t;
    }

    // Noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(unsigned int i = 0; i < observations.size(); i++){

    // Initialize min distance to be a really large number
    double min_distance = std::numeric_limits<double>::max();

    // Initialize the map id to a "null"
    int map_id = -1;

    // Get current observation
    LandmarkObs obs = observations[i];

    for(unsigned int j=0; j<predicted.size(); j++){

      // Get current predicted measurement
      LandmarkObs pred = predicted[j];

      // Calculate distance between observation and predicted measurement
      double distance = dist(obs.x, obs.y, pred.x, pred.y);

      // See if this is a closer neighbor
      if (distance < min_distance) {
        min_distance = distance;
        map_id = pred.id;
      }
    }

    // Assign the nearest predicted landmark to this observation's id
    obs.id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for(int i = 0; i < num_particles; i++){

    Particle curr_particle = particles[i];
    
    vector<LandmarkObs> landmarks_within_range;
    
    // -------------------------------------------
    // Check for valid observations by comparing distance with sensor range
    // -------------------------------------------
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      
      // Get landmark data
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      if (dist(landmark_x, landmark_y, curr_particle.x, curr_particle.y) <= sensor_range) {
        // Save valid landmark within sensor range
        landmarks_within_range.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // -------------------------------------------
    // Transform observations from vehicle coordinates to map coordinates
    // -------------------------------------------
    vector<LandmarkObs> transformed_observations;

    for(unsigned int j = 0; j < observations.size(); j++){
      double transf_x = cos(curr_particle.theta) * observations[j].x - sin(curr_particle.theta) * observations[j].x + curr_particle.x;
      double transf_y = sin(curr_particle.theta) * observations[j].x + cos(curr_particle.theta) * observations[j].y + curr_particle.y;
      transformed_observations.push_back(LandmarkObs{observations[j].id, transf_x, transf_y});
    }

    // -------------------------------------------
    // Association -- Assign the nearest predicted landmark to each observation
    // -------------------------------------------
    dataAssociation(landmarks_within_range, transformed_observations);

    // Reset weight
    curr_particle.weight = 1.0;

    // -------------------------------------------
    // 
    // -------------------------------------------
    for (unsigned int j = 0; j < transformed_observations.size(); j++){

      double obs_x, obs_y, pred_x, pred_y;
      
      int landmark_id = transformed_observations[j].id;

      // Find the predicted associated landmark with the current obervation
      for(unsigned int k = 0; k < landmarks_within_range.size(); k++){
        if (landmarks_within_range[k].id == landmark_id) {
          pred_x = landmarks_within_range[k].x;
          pred_y = landmarks_within_range[k].y;
        }
      }

      obs_x = transformed_observations[j].x;
      obs_y = transformed_observations[j].y;

      // Calculate weight
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];

      // calculate normalization term
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // calculate exponent
      double exponent;
      exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                  + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

      // calculate weight using normalization terms and exponent
      double weight;
      weight = gauss_norm * exp(-exponent);

      // double obs_weight = (1 / (2 * M_PI * std_x * std_y)) * 
      //     exp( -(pow(pred_x - obs_x, 2) / (2*pow(std_x, 2)) + 
      //        (pow(pred_y - obs_y, 2) / (2 * pow(std_y, 2))) ));

      
      // Weight is the product of all observations
      curr_particle.weight *= obs_weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}