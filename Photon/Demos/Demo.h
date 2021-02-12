#pragma once

class Demo
{
public:
	Demo();
	virtual ~Demo();

	virtual void ImGuiRender() {};
	virtual void Update(double interval) {};
	virtual void Render() {};
};

