#include "LPTimer.h"



LPTimer::LPTimer()
{
}


LPTimer::~LPTimer()
{

}

void LPTimer::startTimer()
{
	timer = clock();
}

double LPTimer::getTime()
{
	return ((double)(clock() - timer)) / CLOCKS_PER_SEC;
}

void LPTimer::resetTimer()
{
	startTimer();
}
