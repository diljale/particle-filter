/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	weights.resize(num_particles, 1.0);

	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
	
		Particle particle;
		particle.id = i;		
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);

	}

	is_initialized = true;

	return;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	for (auto& p : particles){

		if (fabs(yaw_rate) > 0.001) {
			p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity/yaw_rate * (cos(p.theta)  - cos(p.theta + yaw_rate * delta_t));
			p.theta  += yaw_rate * delta_t;
		} 
		else {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}

		std::default_random_engine gen;
		std::normal_distribution<double> dist_x(p.x, std_pos[0]);
		std::normal_distribution<double> dist_y(p.y, std_pos[1]);
		std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		
	}

	return;

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	LandmarkObs best_landmark;
	std::vector<LandmarkObs> mapped_observations;
	for (const auto& src : observations){
		
		double min_dist = std::numeric_limits<double>::max(); 

		for (const auto& target : predicted){
			double distance = dist(src.x,src.y,target.x,target.y);
			if (distance < min_dist) {
				min_dist = distance;
				best_landmark = target;
			}
		}
		
		best_landmark.id = src.id;
		mapped_observations.push_back(best_landmark);
	}
	
	observations = mapped_observations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	for(int i=0; i < particles.size(); ++i) {
	    auto& p = particles[i];

	    std::vector<LandmarkObs> observations_absolute;
	    for (const auto& observation: observations){
		  LandmarkObs transformed;
		  transformed.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
		  transformed.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
		  transformed.id = observation.id;
		  observations_absolute.push_back(transformed);
	     }

             std::vector<LandmarkObs> predicted;
	     for (const auto& landmark : map_landmarks.landmark_list){
                  auto distance = dist(p.x,p.y,landmark.x_f,landmark.y_f);
		  if (distance < sensor_range) {
		      LandmarkObs tmp = {landmark.id_i, landmark.x_f, landmark.y_f};
		      predicted.push_back(tmp);		
		  }
	      }
              
	      std::vector<LandmarkObs> observations_actual = observations_absolute;	
	      dataAssociation(predicted, observations_actual);
	      std::vector<int> associations;
	      std::vector<double> sense_x;
	      std::vector<double> sense_y;
	      double probability = 1;		
	      for (int j=0; j < observations_absolute.size(); ++j){
		  double dx = observations_absolute.at(j).x - observations_actual.at(j).x;
		  double dy = observations_absolute.at(j).y - observations_actual.at(j).y;
		  probability *= 1.0/(2*M_PI*sigma_x*sigma_y) * exp(-dx*dx / (2*sigma_x*sigma_x))* exp(-dy*dy / (2*sigma_y*sigma_y));
		  associations.push_back(observations_actual.at(j).id);
		  sense_x.push_back(observations_actual.at(j).x);
		  sense_y.push_back(observations_actual.at(j).y);		    
	      }

	      p = SetAssociations(p, associations, sense_x, sense_y);
	      p.weight = probability;
	      weights[i] = probability;
             
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> dist_p(weights.begin(), weights.end());
	std::vector<Particle> weighted(num_particles);
	std::default_random_engine gen;

	for(int i = 0; i < num_particles; ++i){
		int index = dist_p(gen);
		weighted.at(i) = particles.at(index);
	}

	particles = std::move(weighted);

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	std::cout << v.size() << std::endl;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
