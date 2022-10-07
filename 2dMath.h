#pragma once
#include <cmath>


struct Vec3D
{
	double x;
	double y;
	double z;
};

      

struct Vec2D
        {
            Vec2D(): x(0), y(0) {}
            Vec2D(const double X, const double Y): x(X), y(Y) {}

            void operator+=(const Vec2D &other);
            void operator-=(const Vec2D &other);
            void operator*=(const Vec2D &other);
            void operator/=(const Vec2D &other);

            void operator+=(const double val);
            void operator-=(const double val);
            void operator/=(const double val);
            void operator*=(const double val);

            Vec2D operator+(const Vec2D &other) const;
            Vec2D operator-(const Vec2D &other) const;
            Vec2D operator*(const Vec2D &other) const;
            Vec2D operator/(const Vec2D &other) const;

            Vec2D operator+(const double val) const;
            Vec2D operator-(const double val) const;
            Vec2D operator*(const double val) const;
            Vec2D operator/(const double val) const;

            bool operator==(const Vec2D &other) const;

            void Set(const double X, const double Y);
            Vec2D TurnLeft();
            Vec2D TurnRight();
            Vec2D Normalize();
            bool Similar(const Vec2D other, double precision);
              
			double x;
			double y;
        };


		double FastDist(const Vec2D& v1, const Vec2D& v2);
		double dotProduct(const Vec2D& v1, const Vec2D& v2);
		double Distance(const Vec2D& v1, const Vec2D& v2);
        void CheckBoxBounds(Vec2D & Point, double limit);
        void LimitToCircle(Vec2D & Point, double maxradius);
		double GetAngleFromPoint(const Vec2D& p);
        bool LineVsCircleCollision(const Vec2D & P1, const Vec2D & P2, const Vec2D & C, double radius);
        bool intersection(Vec2D p1, Vec2D p2, Vec2D p3, Vec2D p4, Vec2D & inter);
		int znaminko(double d);
