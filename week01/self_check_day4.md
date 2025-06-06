🧪 PyTorch 微积分自检问卷（对应《动手学深度学习》第 2.4 节）

📘 一、基础概念题（填空 / 选择）
	1.	填空题：导数可以被解释为函数相对于其变量的 __加速度____ 变化率，它也是函数曲线的 __切线__ 的斜率。
	2.	填空题：梯度是一个向量，其分量是多变量函数相对于其所有变量的 __偏导数____。
	3.	选择题：以下关于链式法则的说法正确的是：D
	•	A. 只能用于一元函数的复合
	•	B. 可用于多变量函数的复合
	•	C. 仅适用于线性函数
	•	D. 与偏导数无关
	4.	填空题：在深度学习中，优化问题通常涉及最小化一个 __凹____ 函数，以提高模型的性能。
	5.	选择题：以下关于偏导数的说法正确的是： B
	•	A. 是函数在某一点处的全导数
	•	B. 是函数对某个变量的导数，其他变量保持不变
	•	C. 与梯度无关
	•	D. 只能用于一元函数

🧪 二、代码填空题（基础操作）
6.	填空题：使用 PyTorch 计算函数 f(x) = x^2 在 x = 3 处的导数：
import torch
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)

7.	填空题：计算函数 f(x, y) = x \cdot y 的梯度：
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y
z.backward()
print(x.grad)  # 输出：__3__
print(y.grad)  # 输出：__2__

8.	填空题：验证链式法则：设 y = x^2，z = y^3，计算 \frac{dz}{dx}：
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y ** 3
z.backward()
print(x.grad)  # 输出：__192__

🔍 三、判断与简答题
	9.	判断题：在 PyTorch 中，调用 .backward() 方法后，计算图会被释放，除非指定 retain_graph=True。 (True / False) True
	10.	判断题：梯度下降法是一种利用梯度信息来最小化函数的方法。 (True / False)  True
	11.	简答题：请简要说明链式法则在深度学习中的作用。 
	更多是作用到反向自动微分中，方便计算和调整权重。

💡 四、进阶编程题
	12.	编程题：编写一个函数，接收一个标量张量 x，返回函数 f(x) = \sin(x) \cdot \exp(x) 在 x 处的导数。
    def compute_derivative(x):
    # 请在此处补全代码
		x = torch.tensor(x,requires_grad=True)
		a = sin(x)
		b = exp(x)
		f = a x b
		f.backward()
    return x.grad

    13.	编程题：使用 PyTorch 计算函数 f(x, y) = x^2 + y^2 在点 (3, 4) 处的梯度，并验证其结果。
    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    # 请在此处补全代码
	f = x^2 + y^2
	f.backward()
	print(x.grad)
