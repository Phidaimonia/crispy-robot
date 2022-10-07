#pragma once

#include "2dMath.h"
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>


struct Textures
{
	static void LoadAll();

	static sf::Texture bgTexture;
	static sf::Texture entity;
	static GLuint rect;
	static void drawRectangle();
};


void DrawLine(const Vec2D& v1, const Vec2D& v2, float intensity);
void prepareViewport(int width, int height);