#pragma once
class Timer
{
public:
	Timer();
	virtual ~Timer();

	virtual void startTimer() = 0;
	virtual void resetTimer() = 0;
};

