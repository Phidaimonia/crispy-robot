#include "Entity.h"
#include "2dMath.h"



        bool Entity::isInFoV(const Vec2D& Point, float AngleOffset, float sWide)
        {
            float Dist = Distance(position, Point);
            float NewAng = angle + 3.1415926535 * 0.5 + AngleOffset;

            Vec2D SinCos(cos(NewAng), sin(NewAng));

            NewAng = Distance(SinCos * Dist + position, Point) / Dist * 0.5;

            return asin(NewAng) * 4 < sWide;
        }


        void Entity::reset()
        {
			position.Set(0, 0);
            velocity.Set(0, 0);
			angle = 0;
        }


        void Entity::resetRandPos(double radius)
        {
			position.Set((rand() % 1000 - 500) * 0.001 * radius, (rand() % 1000 - 500) * 0.001 * radius);
			velocity.Set(0.001, 0.001);
			angle = 0;
        }

