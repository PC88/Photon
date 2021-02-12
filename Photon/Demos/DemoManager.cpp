#include "DemoManager.h"
#include "imgui\imgui.h"


DemoManager::DemoManager(Demo*& currentDemoPointer)
	: m_CurrentDemos(currentDemoPointer)
{
}


DemoManager::~DemoManager()
{
}

void DemoManager::ImGuiRender()
{
	for (auto& demo : m_Demos)
	{
		if (ImGui::Button(demo.first.c_str()))
		{
			m_CurrentDemos = demo.second();
		}
	}
}
