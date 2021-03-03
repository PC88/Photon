#pragma once
#include "Demo.h"
class Imgui_DX_12_Demo :
    public Demo
{

	Imgui_DX_12_Demo();
	virtual ~Imgui_DX_12_Demo();

	// Optional inherited functions
	virtual void Update(double interval) override;
	virtual void ImGuiRender() override;
	virtual void Render() override;
};

