🧪 PyTorch 自动微分 自测问卷

参考章节：《动手学深度学习》2.5 自动微分

⸻

📘 一、基础概念题（填空 / 选择）
	1.	填空题：在 PyTorch 中，若要启用某个张量的自动微分功能，应在创建时设置参数 __requires_grad_____ 为 True。
	2.	填空题：当执行 backward() 进行反向传播时，系统会默认计算标量输出对输入的 _第一个？_____ 导数。
	3.	选择题：以下哪个说法关于 .backward() 是正确的？C
		A. .backward() 只能对张量使用，不能对表达式使用
		B. .backward() 一定返回梯度为 1
		C. .backward() 是 PyTorch 中进行自动微分的关键函数
		D. .backward() 总是计算二阶导数
	4.	选择题：若在执行 .backward() 后再次执行 .backward()，但没有保留计算图，会发生什么？ B
		A. 正常执行
		B. 报错，因为计算图已被释放
		C. 第二次梯度值会翻倍
		D. 会自动重新构建图

⸻

🧪 二、代码填空题（掌握基本用法）
	5.	填空题：请补全代码，使其能够计算函数 y = 2 \cdot x^T x 对 x 的梯度。

    import torch

    x = torch.arange(4.0, requires_grad=True)
    y = _2 * x.T * x____________
    y.backward()
    print(x.grad)  # 应输出 tensor([ 0.,  4.,  8., 12.])

    6.	填空题：要保留计算图以便之后计算二阶导数，应在调用 backward() 时设置参数 ___create_graph________ = True。
	7.	填空题：当你对非标量张量调用 .backward()，需要提供一个形状一致的 ___函数？	________ 参数。

🔍 三、判断与简答题
	8.	判断题：只有标量输出才能使用 .backward()。(True / False) False
	9.	简答题：解释什么是“计算图”以及 PyTorch 如何通过它实现自动求导？ 
	计算图会统计每个节点的微分，方便进行反向传播时直接使用进行计算
	10.	简答题：为什么计算高阶导数需要设置 create_graph=True？
		因为要保留计算图，方便计算高阶，否则报错。

⸻

💡 四、进阶编程题
	11.	编程题：使用 PyTorch 自动微分，计算函数 f(x) = x^3 + 2x^2 在 x = 2.0 处的一阶和二阶导数。
	x = torch.? 如何赋值为2
	f = x ** x ** x + 2 * x * x
	f.backward(create_graph=True)
	print(x.grad)
	f.backward()
	print(x.grad)
	12.	编程题：用 torch.autograd.grad() 实现：
给定函数 y = x^2 + 3x，计算在 x = 1.0 处的一阶导数和二阶导数。

⸻

🧠 五、思考题
	13.	思考题：PyTorch 中为什么默认在反向传播后会释放计算图？你在什么情况下会选择保留图？
	14.	思考题：PyTorch 中 .backward() 与 torch.autograd.grad() 有什么区别？分别适用于哪些场景？
