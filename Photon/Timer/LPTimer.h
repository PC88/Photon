#pragma once
#include "Timer\Timer.h"
#include <ctime>
class LPTimer :
	public Timer
{
private:
	clock_t timer;
public:
	LPTimer();
	virtual ~LPTimer();
	virtual void startTimer() override;
	double getTime();
	virtual void resetTimer() override;
};

