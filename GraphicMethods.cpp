#pragma once

#include <SFML/OpenGL.hpp>
#include "2dMath.h"
#include "GraphicMethods.h"

sf::Texture Textures::bgTexture;
sf::Texture Textures::entity;
GLuint Textures::rect;





void Textures::LoadAll()
{
	bgTexture.loadFromFile("textures/background.jpg");
	bgTexture.setRepeated(true);
	bgTexture.setSmooth(true);

	entity.loadFromFile("textures/entity.png");
	entity.setSmooth(true);
	
	rect = glGenLists(1);

	glNewList(rect, GL_COMPILE); 
	glPushMatrix();

	glBegin(GL_TRIANGLES);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0, 1.0);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0, -1.0);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0, 1.0);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0, 1.0);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0, -1.0);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0, -1.0);
	glEnd();

	glPopMatrix();
	glEndList(); //end the list
}


void DrawLine(const Vec2D& v1, const Vec2D& v2, float intensity)
{

	glPushMatrix();

	glLineWidth(intensity);

	glBegin(GL_LINES);
	glVertex2f(v1.x, v1.y);
	glVertex2f(v2.x, v2.y);
	glEnd();

	glPopMatrix();

}


void prepareViewport(int width, int height)
{
	float ratio = width / (float)height;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-ratio, ratio, -1.0f, 1.0f, 1.0f, -1.0f);
	glMatrixMode(GL_MODELVIEW);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
}


void Textures::drawRectangle()
{
	glEnable(GL_TEXTURE_2D);
	glCallList(rect);
	glDisable(GL_TEXTURE_2D);
}


