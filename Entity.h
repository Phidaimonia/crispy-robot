#pragma once

#include "2dMAth.h"
#include "NN.h"
#include "vector"

#define CREATURENUM 30
#define FOODNUM 40
#define SIMULATIONSTEP 1800

#define NNINPCOUNT 14
#define NNOUTCOUNT 8


	struct StatePair
	{
		float score;
		float in[NNINPCOUNT];
		float out[NNOUTCOUNT];
	};

	struct GameReplay
	{
		//int score;
		std::vector<StatePair> timeSteps;
	};



    struct Entity
    {
		Vec2D position;
		Vec2D oldPos;
		Vec2D velocity;

		GameReplay history;


		Vec2D look;
		float angle = 0;
		float sensorAngle = 0;
        int id;
		float score = 0;
		float lastScore = 0;

		NeuralNetwork * brains;

		bool alive = true;

		void reset();
		void resetRandPos(double radius);
		bool isInFoV(const Vec2D& Point, float AngleOffset, float sWide);
    };
