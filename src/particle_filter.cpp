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
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
        particles.resize(num_particles);
	for(auto& p: particles){
	    p.x = dist_x(gen);
	    p.y = dist_y(gen);
	    p.theta = dist_theta(gen);
	    p.weight = 1;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);
	
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

		

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
		
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(auto& obs: observations){
		double min_dist = std::numeric_limits<double>::max();
		for(const auto& pred: predicted){
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if( min_dist > distance){
				min_dist = distance;
				obs.id = pred.id;
			}
		}
	}
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
	
	for(auto&p : particles) {
	   
	    std::vector<LandmarkObs> observations_absolute;
	    for (const auto& observation: observations){
		  LandmarkObs transformed;
		  transformed.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
		  transformed.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
		  observations_absolute.push_back(transformed);
	     }

             std::vector<LandmarkObs> predicted;
	     for (const auto& landmark : map_landmarks.landmark_list){
                  double distance = dist(p.x,p.y,landmark.x_f,landmark.y_f);
		  if (distance < sensor_range) {
		     predicted.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
		  }
	      }
              
	     dataAssociation(predicted, observations_absolute);
		
	      p.weight = 1.0;		
	      for(const auto& obs : observations_absolute){
		      auto landmark = map_landmarks.landmark_list.at(obs.id-1);
		      double dx = pow(obs.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
		      double dy = pow(obs.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
		      double w = exp(-(dx + dy)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		      p.weight *=  w;
	    }
            weights.push_back(p.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<> dist_p(weights.begin(), weights.end());
	std::vector<Particle> weighted(num_particles);
	std::random_device rd;
        std::mt19937 gen(rd());

	for(int i = 0; i < num_particles; ++i){
		int index = dist_p(gen);
		weighted[i] = particles[index];
	}

	particles = weighted;
	weights.clear();

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
