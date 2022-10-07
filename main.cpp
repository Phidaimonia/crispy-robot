#pragma once

#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <math.h>
#include <windows.h>
#include "GraphicMethods.h"
#include "Level.h"
//#include "NN.h"

#include <iostream>
#include <random>
#include <sstream>
#include <algorithm> 
#include <algorithm> 
#include <mutex>

using namespace std;

void render();

unsigned int wWidth = 1920;
unsigned int wHeight = 1080;

unsigned int mX = 0;
unsigned int mY = 0;

Vec2D cameraPos = Vec2D(0, 0);
float cameraZoom = 1.0;
bool keys[256] = { false };


std::mutex mut;

bool running = true;
bool paused = false;
bool BGtraining = true;

sf::Text text;
sf::Font font;


Level level;
Level evo;
NeuralNetwork tmpExpNN;


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

bool sortFunc(Entity i, Entity j) 
{ 
	return (i.score < j.score); 
}


void ProcessEvents(sf::Window & window)
{
	sf::Event event;
	double mxReward = -1000;
	int ind = 0;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
			/////////////////////////////////////////////////
		case sf::Event::KeyPressed:
			keys[event.key.code] = true;
			switch (event.key.code)
			{
			case sf::Keyboard::Escape:
				running = false;
				break;

			case sf::Keyboard::E:
				level.wantedReward -= 0.1;
				break;

			case sf::Keyboard::R:
				level.wantedReward += 0.1;
				break;


			case sf::Keyboard::F:
				BGtraining = !BGtraining;
				break;
				

			case sf::Keyboard::Q:
				mut.lock();
				for(int i = 0; i < evo.entity.size(); ++i)
					if (evo.entity[i].score > mxReward)
					{
						mxReward = evo.entity[i].score;
						ind = i;
					}

				for (int i = 0; i < level.entity.size(); ++i)
						level.entity[i].brains->CopyFrom(evo.entity[ind].brains);
				mxReward = -1000;
				mut.unlock();
				break;

			case sf::Keyboard::Space:
				paused = !paused;
				break;
			}	break;

			case sf::Event::KeyReleased:
				keys[event.key.code] = false;
			/////////////////////////////////////////////////

		case sf::Event::MouseWheelScrolled:
			cameraZoom += event.mouseWheelScroll.delta * 0.1 * cameraZoom;
			break;


		case sf::Event::MouseMoved:
			mX = event.mouseMove.x;
			mY = event.mouseMove.y;
			break;


		case sf::Event::Closed:
			running = false;
			break;

		case sf::Event::MouseButtonPressed:
			//
			break;

		case sf::Event::MouseButtonReleased:
			break;

		default:
			break;
		}
	}
}


void updatePlayer(int index)
{
	Vec2D keyboardVelocity = Vec2D(0, 0);

	if (keys[22]) keyboardVelocity += Vec2D(0.0, 1.0); // W
	if (keys[18]) keyboardVelocity += Vec2D(0.0, -1.0); // S
	if (keys[0]) keyboardVelocity += Vec2D(-1.0, 0.0); // A
	if (keys[3]) keyboardVelocity += Vec2D(1.0, 0.0); // D

	if (abs(keyboardVelocity.x) == abs(keyboardVelocity.y)) keyboardVelocity * 0.5 * sqrt(2.0);
	level.entity[index].velocity = keyboardVelocity * 0.006f;

	level.entity[index].position += level.entity[index].velocity * level.frameTime;
	LimitToCircle(level.entity[index].position, level.cageSize);
	cameraPos += (level.entity[index].position - cameraPos) * 0.1;

	level.entity[index].oldPos = level.entity[index].position;
}


void testInputs()
{
	float input[5];

	for (int s = 0; s < 5; ++s) input[s] = 0;  
	input[0] = FastDist(level.entity[0].velocity, Vec2D(0, 0)) * 10000;
	if (input[0] > 10) input[0] = 10;

	input[1] = level.entity[0].sensorAngle;
	input[2] = distToBoxEdges(level.entity[0].position, level.cageSize);
	input[3] = nearFood(level.food, level.entity[0]); 
	input[4] = foodInFoV(level.food, level.entity[0]); 

	std::ostringstream oss;

	for (size_t i = 0; i < 5; ++i)
		oss << input[i] << std::endl;

	oss << "Background evolution running:   " << BGtraining << std::endl;
	oss << "Wanted reward:    " << level.wantedReward << std::endl;
	text.setString(oss.str());
}


void drawBG()
{
	glEnable(GL_TEXTURE_2D);
	glPushMatrix();
	glScaled(100.0, 100.0, 0.0);
	sf::Texture::bind(&Textures::bgTexture);
	Textures::drawRectangle();
	glPopMatrix();
	glDisable(GL_TEXTURE_2D);
}


void render()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPushMatrix();
	glScalef(cameraZoom, cameraZoom, 1.0);
	glTranslatef(-cameraPos.x, -cameraPos.y, 0);

	drawBG();
	level.renderMap();
	level.renderEntities();

	//DrawLine(Vec2D(-0.1, 0.0), Vec2D(0.1, 0.0), 4.0f);
	//DrawLine(Vec2D(0.0, -0.1), Vec2D(0.0, 0.1), 4.0f);

	glPopMatrix();
	glDisable(GL_BLEND);
}


void testNN()
{
//     Neural net learning test
			std::vector<size_t> layout = {160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 6};
			NeuralNetwork LearnTest = NeuralNetwork(layout, 2);
			LearnTest.Randomize(1.0);

			float input[2];
			float desiredoutput[6];
			float output[6];

			//LearnTest.Layer[0].neuron[0].Weights[0] = 2.0;
			//LearnTest.Layer[0].neuron[0].BiasWeight = 0;

			//LearnTest.Layer[1].neuron[0].Weights[0] = 1.2;
			//LearnTest.Layer[1].neuron[0].BiasWeight = 0.2;

		
			for(int i = 0; i < 100000; ++i)
			{
				input[0] = i % 4 / 2;
				input[1] = i % 3 / 2;

				desiredoutput[0] = (input[0] == 1 ^ input[1] == 1) * 0.5 + 0.1;
				desiredoutput[1] = (input[0] == 1 | input[1] == 1);
				desiredoutput[2] = (input[0] == 1 & input[1] == 1);
				desiredoutput[3] = !(input[0] == 1 ^ input[1] == 1);
				desiredoutput[4] = !(input[0] == 1 | input[1] == 1);
				desiredoutput[5] = !(input[0] == 1 & input[1] == 1);


				float er = LearnTest.Learn(input, desiredoutput, 0.01);
		
				//float er = LearnTest.LearnLayer(input, desiredoutput, 0, 0.00001);
				//LearnTest.LearnLayer(input, desiredoutput, 4, 0.01);

				//std::cout << "MSE:      " << er << std::endl;

				input[0] = 1.0;
				input[1] = 0.0;

				LearnTest.Compute(input);
				memcpy(output, LearnTest.Layer[LearnTest.LayerCount - 1].Output, sizeof(float) * 6);

				

				//std::cout << "Input: " << input[0] << ' ' << input[1] << std::endl;
				//std::cout << "Should be: " << 1 << ' ' << 1 << ' ' << 0 << ' ' << 0 << ' ' << 0 << ' ' << 1 << std::endl;
				std::cout << i << ":  Result: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << ' ' << output[4] << ' ' << output[5] << std::endl << std::endl;

			} 

			for (int i = 0; i < 5; ++i)
			{
				input[0] = i % 4 / 2;
				input[1] = i % 3 / 2;

				desiredoutput[0] = (input[0] == 1 ^ input[1] == 1);
				desiredoutput[1] = (input[0] == 1 | input[1] == 1);
				desiredoutput[2] = (input[0] == 1 & input[1] == 1);
				desiredoutput[3] = !(input[0] == 1 ^ input[1] == 1);
				desiredoutput[4] = !(input[0] == 1 | input[1] == 1);
				desiredoutput[5] = !(input[0] == 1 & input[1] == 1);

				LearnTest.Compute(input);
				memcpy(output, LearnTest.Layer[LearnTest.LayerCount - 1].Output, sizeof(float) * 6);

				std::cout << "Input: " << input[0] << ' ' << input[1] << std::endl;
				std::cout << "Should be: " << desiredoutput[0] << ' ' << desiredoutput[1] << ' ' << desiredoutput[2] << ' ' << desiredoutput[3] << ' ' << desiredoutput[4] << ' ' << desiredoutput[5] << std::endl;
				std::cout << "Result: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << ' ' << output[4] << ' ' << output[5] << std::endl << std::endl;
			} 




			desiredoutput[0] = 0.145001f;
			desiredoutput[1] = 0.833001f;
			desiredoutput[2] = 0.399001f;
			desiredoutput[3] = 0.222001f;
			desiredoutput[4] = 0.090011f;
			desiredoutput[5] = 0.771119f;

			for (int i = 0; i < 100000; ++i)
			{
				input[0] = i % 4 / 2;
				input[1] = i % 3 / 2;

				LearnTest.Learn(input, desiredoutput, 0.1);
			}


			std::cout << "Only learn biases: " << endl;

			for (int i = 0; i < 7; ++i)
			{
				input[0] = i % 4 / 2;
				input[1] = i % 3 / 2;

				LearnTest.Compute(input);
				memcpy(output, LearnTest.Layer[LearnTest.LayerCount - 1].Output, sizeof(float) * 6);

				//std::cout << "Input: " << input[0] << ' ' << input[1] << std::endl;
				std::cout << "Should: " << desiredoutput[0] << ' ' << desiredoutput[1] << ' ' << desiredoutput[2] << ' ' << desiredoutput[3] << ' ' << desiredoutput[4] << ' ' << desiredoutput[5] << std::endl;
				std::cout << "Result: " << output[0] << ' ' << output[1] << ' ' << output[2] << ' ' << output[3] << ' ' << output[4] << ' ' << output[5] << std::endl << std::endl;
			} 
}



void createExperienceNN()
{
	std::vector<size_t> layout = { 8, 8, NNOUTCOUNT };
	level.expNN.Reshape(layout, NNINPCOUNT + 1);
	level.expNN.Randomize(1.0);

	tmpExpNN.Reshape(layout, NNINPCOUNT + 1);
	tmpExpNN.Randomize(1.0);
}


void evolutionThreadLoop(sf::Window* window)
{
	evo.prepare(CREATURENUM);
	evo.reset();
	evo.saveExperience = false;


	while (window->isOpen() && running)
	{
		for (size_t t = 0; t < SIMULATIONSTEP; ++t)  
		{
			if (!running) return;
			evo.updateAll(16.66666666, 0);
		}

		// sort by score
		sort(evo.entity.begin(), evo.entity.end(), sortFunc); 

		/*double overallMSE = 0;

		float inp[NNINPCOUNT + 1];
		memset(inp, 0, sizeof(float) * (NNINPCOUNT + 1));

		if (BGtraining)
		for (int o = 0; o < 4; ++o)
		for (size_t i = 0; i < CREATURENUM; ++i)
		{
			//mut.try_lock();
			if (evo.entity[i].history.timeSteps.size() > 6)
				for (size_t t = 4; t < evo.entity[i].history.timeSteps.size()-1; ++t)
				{
					memcpy(inp, evo.entity[i].history.timeSteps[t].in, sizeof(float) * NNINPCOUNT);
					inp[NNINPCOUNT] = evo.entity[i].history.timeSteps[t].score - evo.entity[i].history.timeSteps[t-1].score;

				
					overallMSE += tmpExpNN.Learn(inp, evo.entity[i].history.timeSteps[t].out, 0.0001f);

					//cout << "MSE: " << inp[NNINPCOUNT] << "\n";
				}
			//mut.unlock();
		}  

		//for (int i = 2; i < min(evo.entity.size(), level.entity.size()); ++i)        
		//	level.entity[i].brains->CopyFrom(evo.entity[i].brains);

		//tmpExpNN

		mut.lock();
		level.expNN.CopyFrom(&tmpExpNN);
		mut.unlock();

		cout << "MSE: " << overallMSE / (CREATURENUM * SIMULATIONSTEP)  << "\n"; */





		mut.lock();
		// change population
		//for (size_t i = 1; i < CREATURENUM * 0.2; ++i)
		//	evo.entity[i].brains->Randomize(10.0);

		for (size_t i = 0; i < CREATURENUM * 0.6; ++i)
			evo.entity[CREATURENUM - rand() % 10 - 1].brains->Mutate(0.01, evo.entity[i].brains);
		mut.unlock();


		evo.generation++;
		cout << "Generation " << evo.generation << "		Best score " << evo.entity[evo.entity.size() - 1].score << "\n";

		//if(!(evo.generation % 10))
		//		evo.entity[CREATURENUM-1].brains->SaveToFile("gen" + evo.generation);  // error

		evo.reset();
	}
}


int main()
{
	//testNN();
	//string s;
	//cin >> s;


	sf::RenderWindow window(sf::VideoMode(), "Evolution game", sf::Style::Fullscreen);

	font.loadFromFile("textures/arial.ttf");
	text.setFont(font);
	text.setString("Info:");
	text.setCharacterSize(16);
	text.setFillColor(sf::Color::Black);
	text.setStyle(sf::Text::Regular); 

	window.setActive(false);  
	window.setMouseCursorVisible(false);
	window.setActive(true);
	window.setFramerateLimit(60);

	createExperienceNN();

	Textures::LoadAll();
	level.prepare(5);  //        CREATURE COUNT
	level.reset();
	level.saveExperience = false;

	sf::Thread thread(&evolutionThreadLoop, &window);
	thread.launch();

	while (window.isOpen() && running)
	{
		ProcessEvents(window);

		mut.lock();
		if (!paused) level.updateAll(16.66666666f, 1);
		updatePlayer(0);
		mut.unlock();

		prepareViewport(wWidth, wHeight);
		render();

		testInputs();
		window.pushGLStates();
		window.draw(text);
		window.popGLStates();
		window.display();
	} 
}


