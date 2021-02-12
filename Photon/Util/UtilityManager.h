#pragma once

#include <memory>
#include <iostream>
#include <string>

class UtilityManager
{
public:
	static UtilityManager& instance();


	UtilityManager(UtilityManager const&) = delete;
	void operator=(UtilityManager const&) = delete;

private:
	UtilityManager();
	virtual ~UtilityManager();
};

