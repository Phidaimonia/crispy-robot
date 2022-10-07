#include "2dMath.h"

            void Vec2D::operator+=(const Vec2D &other)
            {
                x += other.x;
                y += other.y;
            }

            void Vec2D::operator-=(const Vec2D &other)
            {
                x -= other.x;
                y -= other.y;
            }

            void Vec2D::operator*=(const Vec2D &other)
            {
                x *= other.x;
                y *= other.y;
            }

            void Vec2D::operator/=(const Vec2D &other)
            {
                x /= other.x;
                y /= other.y;
            }

            void Vec2D::operator+=(const double val)
            {
                x += val;
                y += val;
            }

            void Vec2D::operator-=(const double val)
            {
                x -= val;
                y -= val;
            }

            void Vec2D::operator/=(const double val)
            {
                x /= val;
                y /= val;
            }

            void Vec2D::operator*=(const double val)
            {
                x *= val;
                y *= val;
            }

            Vec2D Vec2D::operator+(const Vec2D &other) const
            {
                return Vec2D(x + other.x, y + other.y);
            }

            Vec2D Vec2D::operator-(const Vec2D &other) const
            {
                return Vec2D(x - other.x, y - other.y);
            }

            Vec2D Vec2D::operator*(const Vec2D &other) const
            {
                return Vec2D(x * other.x, y * other.y);
            }

            Vec2D Vec2D::operator/(const Vec2D &other) const
            {
                return Vec2D(x / other.x, y / other.y);
            }

            Vec2D Vec2D::operator+(const double val) const
            {
                return Vec2D(x + val, y + val);
            }

            Vec2D Vec2D::operator-(const double val) const
            {
                return Vec2D(x - val, y - val);
            }

            Vec2D Vec2D::operator*(const double val) const
            {
                return Vec2D(x * val, y * val);
            }

            Vec2D Vec2D::operator/(const double val) const
            {
                return Vec2D(x / val, y / val);
            }

            void Vec2D::Set(const double X, const double Y)
            {
                x = X;
                y = Y;
            }

            Vec2D Vec2D::Normalize()
            {
                double dist = sqrt(x * x + y * y);
                return Vec2D(x / dist, y / dist);
            }

            bool Vec2D::operator==(const Vec2D &other) const
            {
                return ((x == other.x) && (y == other.y));
            }

            bool Vec2D::Similar(const Vec2D other, double precision)
            {
                Vec2D Diff = (*this) - other;
                return ((abs(Diff.x) < precision) && (abs(Diff.y) < precision));
            }





		double FastDist(const Vec2D& v1, const Vec2D& v2)
        {
			double XDist = v1.x - v2.x;
			double YDist = v1.y - v2.y;
            return XDist * XDist + YDist * YDist;
        }



		double Distance(const Vec2D& v1, const Vec2D& v2)
        {
            return sqrt(FastDist(v1, v2));
        }


        void CheckBoxBounds(Vec2D & Point, double limit)
        {
            if (Point.x > limit) Point.x = limit;
            if (Point.x < -limit) Point.x = -limit;
            if (Point.y > limit) Point.y = limit;
            if (Point.y < -limit) Point.y = -limit;
        }


        void LimitToCircle(Vec2D & Point, double maxradius)
        {
			double dist_sqr = Point.x*Point.x + Point.y*Point.y;
            if(dist_sqr > pow(maxradius, 2))
            {
                Vec2D dir = Point.Normalize();
                Point = dir * maxradius;
            }
        }


		double GetAngleFromPoint(const Vec2D& p)
        {
            return 3.14159265358f / 2.0 - atan2(p.x, p.y);
        }


        bool LineVsCircleCollision(const Vec2D & P1, const Vec2D & P2, const Vec2D & C, double radius)
        {
            Vec2D D1 = P1 - C;
            Vec2D tmp = P2 - P1;

           // if(Distance(P1, C) < radius) return true;
           // if(Distance(P2, C) < radius) return true;

			double a = tmp.x * tmp.x + tmp.y * tmp.y;
			double b = 2 * (tmp.x * D1.x + tmp.y * D1.y);
			double c = (D1.x * D1.x) + (D1.y * D1.y) - radius * radius;
			double delta = b * b - 4 * a * c;

            if (delta >= 0) return true;
            return false;
        }



		double dotProduct(const Vec2D& v1, const Vec2D& v2)
		{
			return v1.x * v2.x + v1.y * v2.y;
		}


        Vec2D Vec2D::TurnLeft()
        {
            return Vec2D(-y, x);
        }


        Vec2D Vec2D::TurnRight()
        {
            return Vec2D(y, -x);
        }



       



		bool intersection(Vec2D p1, Vec2D p2, Vec2D p3, Vec2D p4, Vec2D & inter) {
			if (p1.x == p2.x) p1.x += 0.00001;
			if (p1.y == p2.y) p1.y += 0.00001;
			if (p3.x == p4.x) p3.x += 0.00001;
			if (p3.y == p4.y) p3.y += 0.00001;

			// Components of the first segment's rect.
			Vec2D v1 = Vec2D(p2.x - p1.x, p2.y - p1.y); // Directional vector
			double a1 = v1.y;
			double b1 = -v1.x;
			double c1 = v1.x * p1.y - v1.y * p1.x;

			// Components of the second segment's rect.
			Vec2D v2 = Vec2D(p4.x - p3.x, p4.y - p3.y);
			double a2 = v2.y;
			double b2 = -v2.x;
			double c2 = v2.x * p3.y - v2.y * p3.x;

			// Calc intersection between RECTS.
			Vec2D intersection;
			double det = a1 * b2 - b1 * a2;
			if (det != 0) intersection = Vec2D((b2 * (-c1) - b1 * (-c2)) / det, (a1 * (-c2) - a2 * (-c1)) / det);

			//if (det != 0) return true;

			inter = intersection;

			// Checks ff segments collides.
			if(		((p1.x <= intersection.x && intersection.x <= p2.x) || (p2.x <= intersection.x && intersection.x <= p1.x)) &&
					((p1.y <= intersection.y && intersection.y <= p2.y) || (p2.y <= intersection.y && intersection.y <= p1.y)) &&
					((p3.x <= intersection.x && intersection.x <= p4.x) || (p4.x <= intersection.x && intersection.x <= p3.x)) &&
					((p3.y <= intersection.y && intersection.y <= p4.y) || (p4.y <= intersection.y && intersection.y <= p3.y)))
				return true;

			return false;
		};


		int znaminko(double d)
		{
			return d < 0.0 ? -1 : 1;
		}



