#include "UtilityManager.h"

UtilityManager& UtilityManager::instance()
{
	static UtilityManager _self;
	return _self;
}

UtilityManager::UtilityManager()
{

}

UtilityManager::~UtilityManager()
{

}