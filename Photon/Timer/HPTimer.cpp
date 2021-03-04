#include "HPTimer.h"



HPTimer::HPTimer()
{
}


HPTimer::~HPTimer()
{
}

void HPTimer::startTimer()
{
	m_Start = std::chrono::steady_clock::now();
}

void HPTimer::resetTimer()
{
	startTimer();
}

std::chrono::milliseconds HPTimer::getTime()
{
	m_Timer = std::chrono::duration_cast<std::chrono::milliseconds>(m_End - m_Start);
	return m_Timer;
}