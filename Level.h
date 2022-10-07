#pragma once

#include "2dMath.h"
#include "Entity.h"
#include <vector>
#include "NN.h"
#include <SFML/OpenGL.hpp>


struct Level
{
	double frameTime;
	double worldUptime = 0.0;
	size_t generation = 0;

	bool saveExperience = false;

	const double cageSize = 15;

	double wantedReward = 5;

	std::vector<Entity> entity;
	Vec2D food[FOODNUM];

	NeuralNetwork expNN;

	void prepare(size_t num);

	void computeInputs(size_t startindex);
	void processOutputs(size_t startindex);

	void updateAll(double frametime, int startFrom);
	void updateScore();

	void reset();

	void renderEntities();
	void renderMap();
};


float dangerEntity(std::vector<Entity>& ent, Entity e);
float entityInFoV(std::vector<Entity>& ent, Entity e);
float distToBoxEdges(Vec2D pos, float size);
float foodInFoV(const Vec2D * food, Entity e);
float nearFood(const Vec2D * food, Entity e);