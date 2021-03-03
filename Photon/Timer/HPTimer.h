#pragma once
#include <chrono>
#include "Timer/Timer.h"

class HPTimer : public Timer
{
public:
	HPTimer();
	~HPTimer();

	virtual void startTimer() override;
	virtual void resetTimer() override;
	std::chrono::milliseconds getTime();
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_Start;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_End;
	std::chrono::milliseconds m_Timer;
};

