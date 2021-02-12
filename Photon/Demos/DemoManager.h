#pragma once
#include "Demo.h"
#include <string>
#include <functional>
#include <Vector>

class DemoManager :
	public Demo
{
public:

	DemoManager(Demo*& currentDemoPointer);
	virtual ~DemoManager();

	void ImGuiRender() override;

	template<typename T>
	void RegisterDemo(const std::string& name)
	{
		m_Demos.push_back(std::make_pair(name, []() {return new T(); }));
	}

private:
	// this is used so owner ship/creation is managed by a lambda on construction - instead of holding an already existing object ref -PC
	std::vector<std::pair<std::string, std::function<Demo*()>>> m_Demos;
	Demo*& m_CurrentDemos;
};

