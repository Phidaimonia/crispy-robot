#include "Level.h"
#include "GraphicMethods.h"
#include <iostream>



void Level::prepare(size_t num)
{
	for (size_t i = 0; i < num; ++i)
	{
		Entity tmp;
		tmp.angle = 0;
		tmp.sensorAngle = 0.7;
		tmp.id = i;
		tmp.resetRandPos(20);
		tmp.score = 0;
		tmp.alive = true;
		entity.push_back(tmp);
	}

	for (size_t i = 0; i < num; ++i)
	{
		entity[i].brains = new NeuralNetwork({8, NNOUTCOUNT }, NNINPCOUNT);
		entity[i].brains->Randomize(10.0);
		entity[i].history.timeSteps.reserve(SIMULATIONSTEP);
		//brains[i]->ReLoadFromFile("brain.txt");
	}


	for (size_t i = 0; i < FOODNUM; ++i)
	{
		food[i] = Vec2D((rand() % 1000 - 500) * 0.001 * cageSize, (rand() % 1000 - 500) * 0.001 * cageSize);
	}
}


float dangerEntity(std::vector<Entity>& ent, Entity e)
{
	double distKoef = 100;  // max
	for (size_t i = 0; i < ent.size(); ++i)
		if(e.id != ent[i].id)
			if (ent[i].alive) distKoef = std::min(FastDist(e.position, ent[i].position), distKoef);

	return std::min(1.0 / distKoef, 1.0);
}


float nearFood(const Vec2D * food, Entity e)
{
	double distKoef = 100;  // max
	for (size_t i = 0; i < FOODNUM; ++i)
			distKoef = std::min(FastDist(e.position, food[i]), distKoef);

	return std::min(1.0 / distKoef, 1.0);
}


float foodInFoV(const Vec2D * food, Entity e)
{
	//int index = -1;
	double dist = 99999;
	for (int i = 0; i < FOODNUM; ++i)
			if (e.isInFoV(food[i], -1.570746, e.sensorAngle))
			{
				double d = Distance(food[i], e.position);
				if (d < dist)
				{
					dist = d;
					//index = i;
				}
			}
	return std::min(1.0 / dist, 1.0);
}



float entityInFoV(std::vector<Entity>& ent, Entity e)
{
	//int index = -1;
	double dist = 99999;
	for (int i = 0; i < ent.size(); ++i)
		if (e.id != ent[i].id)
				if (e.isInFoV(ent[i].position, -1.570746, e.sensorAngle))
				{
					double d = Distance(ent[i].position, e.position);
					if (d < dist)
					{
						dist = d;
						//index = i;
					}
				}
	return std::min(1.0 / dist, 1.0);
}


float distToBoxEdges(Vec2D pos, float size)
{
	double distKoef = std::min(abs(abs(pos.x) - size), abs(abs(pos.y) - size));
	return std::min(1.0 / distKoef, 1.0);
}



void Level::reset()
{
	worldUptime = 0.0;

	for (size_t i = 0; i < entity.size(); ++i)
	{
		entity[i].angle = 0;
		entity[i].sensorAngle = 0.7;
		entity[i].resetRandPos(20);
		entity[i].alive = true;
		entity[i].score = 0;
//		entity[i].history.score = 0;
		entity[i].history.timeSteps.clear();
		entity[i].brains->Reset();
	}
}



void Level::computeInputs(size_t startindex)
{
	float input[NNINPCOUNT];

	for (size_t i = startindex; i < entity.size(); ++i)   // index 0 is the player
	{
		if (!entity[i].alive) continue;

		memset(input, 0, sizeof(float) * NNINPCOUNT);

		input[0] = FastDist(entity[i].velocity, Vec2D(0, 0)) * 1000;  // speed
		if (input[0] > 1.0) input[0] = 1.0;
		if (input[0] < -1.0) input[0] = -1.0;

		input[1] = entity[i].sensorAngle * 0.6666;
		if (input[1] > 1.0) input[1] = 1.0;
		if (input[1] < -1.0) input[1] = -1.0;

		input[2] = distToBoxEdges(entity[i].position, cageSize);
		if (input[2] > 1.0) input[2] = 1.0;
		if (input[2] < -1.0) input[2] = -1.0;


		int l = entity[i].brains->LayerCount - 1;
		input[3] = entity[i].brains->Layer[l].Output[0];  //  otocka +/-
		input[4] = entity[i].brains->Layer[l].Output[1];  // 
		input[5] = entity[i].brains->Layer[l].Output[2];  //
		input[6] = entity[i].brains->Layer[l].Output[3];  //
		input[7] = entity[i].brains->Layer[l].Output[4];  //  otocka +/-
		input[8] = entity[i].brains->Layer[l].Output[5];  // 
		input[9] = entity[i].brains->Layer[l].Output[6];  //
		input[10] = entity[i].brains->Layer[l].Output[7];


		/*input[7] = sin(worldUptime * 0.01);			// current goal - 
		input[8] = sin(worldUptime * 0.001);
		input[9] = sin(worldUptime * 0.0001);
		input[10] = cos(worldUptime * 0.01);
		input[11] = cos(worldUptime * 0.001);
		input[12] = cos(worldUptime * 0.0001);    */

		input[11] = nearFood(food, entity[i]);
		if (input[11] > 1.0) input[11] = 1.0;
		if (input[11] < -1.0) input[11] = -1.0;

		input[12] = foodInFoV(food, entity[i]);
		if (input[12] > 1.0) input[12] = 1.0;
		if (input[12] < -1.0) input[12] = -1.0;

		input[13] = (entity[i].score - entity[i].lastScore) > 2;





		entity[i].brains->Compute(input);





		/*if(!saveExperience)
		if (i == 1)
		{
			float tmpp[NNINPCOUNT + 1];
			memset(tmpp, 0, sizeof(float) * (NNINPCOUNT + 1));

			memcpy(tmpp, input, sizeof(float) * NNINPCOUNT);
			tmpp[NNINPCOUNT] = wantedReward;
			expNN.Compute(tmpp);

			memcpy(entity[i].brains->Layer[l].Output, expNN.Layer[expNN.LayerCount-1].Output, sizeof(float) * NNOUTCOUNT);

			//std::cout << "Expected reward: " << expNN.Layer[expNN.LayerCount - 1].Output[0] << std::endl;
		} */



		if (saveExperience)
		{
			StatePair tmp;
			
			memcpy(tmp.in, input, sizeof(float) * NNINPCOUNT);
			memcpy(tmp.out, entity[i].brains->Layer[l].Output, sizeof(float) * NNOUTCOUNT);
			//tmp.score = entity[i].score;   // ne

			entity[i].history.timeSteps.push_back(tmp);
			size_t index = entity[i].history.timeSteps.size() - 1;
			if(index >= 0)
				entity[i].history.timeSteps[index].score = entity[i].score;

		}
	}

}


void Level::processOutputs(size_t startindex)
{
	for (size_t i = startindex; i < entity.size(); ++i)   // index 0 is the player
	{
		if (!entity[i].alive) continue;

		int l = entity[i].brains->LayerCount - 1;

		entity[i].oldPos = entity[i].position;

		//entity[i].score -= abs((entity[i].brains->Layer[l].Output[0] * 2.0 - 1.0) * 0.01 * frameTime);

		entity[i].angle += (entity[i].brains->Layer[l].Output[0] * 2.0 - 1.0) * 0.01 * frameTime;
		if (entity[i].angle > 6.28318537) entity[i].angle = 0;
		if (entity[i].angle < 0) entity[i].angle = 6.28318537;

		entity[i].sensorAngle += (entity[i].brains->Layer[l].Output[1] * 2.0 - 1.0) * 0.002 * frameTime;
		if (entity[i].sensorAngle > 1.5) entity[i].sensorAngle = 1.5;
		if (entity[i].sensorAngle < 0.1) entity[i].sensorAngle = 0.1;


		Vec2D look = Vec2D(cos(entity[i].angle), sin(entity[i].angle));
		entity[i].velocity += look * 0.001 * (entity[i].brains->Layer[l].Output[2] * 2.0 - 1.0);  // pohyb rovne

		look = look.TurnLeft();
		entity[i].velocity += look * 0.001 * (entity[i].brains->Layer[l].Output[3] * 2.0 - 1.0);  // pohyb do strany
		entity[i].velocity = entity[i].velocity * 0.9;

		LimitToCircle(entity[i].velocity, 0.5);

		entity[i].position += entity[i].velocity * frameTime;
		LimitToCircle(entity[i].position, cageSize);
	}
}



void Level::updateAll(double frametime, int startFrom)
{
	frameTime = frametime;
	worldUptime += frametime;

	computeInputs(startFrom);
	processOutputs(startFrom);
	updateScore();
}




void Level::updateScore()
{
	for (size_t i = 0; i < entity.size(); ++i)   // index 0 is the player
	{
		if (!entity[i].alive) continue;
		entity[i].lastScore = entity[i].score;
		entity[i].score -= FastDist(Vec2D(0, 0), entity[i].velocity) * 10;
		entity[i].score -= 0.002;

		for (int f = 0; f < FOODNUM; ++f)
		{
			if (FastDist(food[f], entity[i].position) < 0.15)
			{
				//entity[i].lastScore = entity[i].score;
				entity[i].score += 5;
				food[f] = Vec2D((rand() % 1000 - 500) * 0.001 * cageSize, (rand() % 1000 - 500) * 0.001 * cageSize);
			}
		}


		/*for (size_t j = i; j < entity.size(); ++j)   // index 0 is the player
		{
			if (entity[i].id != entity[j].id)
				if (FastDist(entity[i].position, entity[j].position) < 0.03)
				{
					Vec2D center = (entity[i].position + entity[j].position) * 0.5;
					entity[i].velocity = (entity[i].position - center).Normalize() * 0.04;
					entity[j].velocity = (entity[j].position - center).Normalize() * 0.04;

					float s1 = entity[i].velocity.x + entity[i].velocity.y;
					float s2 = entity[j].velocity.x + entity[j].velocity.y;

					//if (s1 > s2) entity[i].score++;
					//else  entity[j].score++;
				}
		} */
	}
}




void Level::renderMap()
{
	DrawLine(Vec2D(-cageSize, -cageSize), Vec2D(-cageSize, cageSize), 4.0f);
	DrawLine(Vec2D(-cageSize, cageSize), Vec2D(cageSize, cageSize), 4.0f);
	DrawLine(Vec2D(cageSize, cageSize), Vec2D(cageSize, -cageSize), 4.0f);
	DrawLine(Vec2D(-cageSize, -cageSize), Vec2D(cageSize, -cageSize), 4.0f);
}



void Level::renderEntities()
{
	for (int f = 0; f < FOODNUM; ++f)
	{
		glPushMatrix();
		glColor3f(0.0f, 1.0f, 0.5f);
		sf::Texture::bind(&Textures::entity);

		glTranslatef(food[f].x, food[f].y, 0);
		glScalef(0.1f, 0.1f, 1.0f);

		Textures::drawRectangle();
		glColor3f(1.0f, 1.0f, 1.0f);

		glPopMatrix();
	}

		for(int i = 0; i < entity.size(); ++i)
		{
			Entity & p = entity[i];
			if (!p.alive) continue;
			
			glPushMatrix();
			sf::Texture::bind(&Textures::entity);

			Vec2D p1 = Vec2D(cos(p.angle - p.sensorAngle * 0.5) * 2.0, sin(p.angle - p.sensorAngle * 0.5) * 2.0);
			Vec2D p2 = Vec2D(cos(p.angle + p.sensorAngle * 0.5) * 2.0, sin(p.angle + p.sensorAngle * 0.5) * 2.0);
			DrawLine(p.position, p.position + p1, 4);
			DrawLine(p.position, p.position + p2, 4);

			glTranslatef(p.position.x, p.position.y, 0);
			glScalef(0.13f, 0.13f, 1.0f);  // nemenit
			glRotatef(p.angle * 57.2957795132f - 90, 0.0f, 0.0f, 1.0f);

			if (i == 1) glColor3f(0.0f, 1.0f, 0.5f);
			Textures::drawRectangle();
			glColor3f(1.0f, 1.0f, 1.0f);

			glPopMatrix();
		}		
}
