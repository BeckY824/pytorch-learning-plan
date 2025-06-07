🧪 PyTorch 编程基础自测题（从易到难）

📚 适用范围：刚学习 PyTorch 编程的开发者 / 学生
📘 参考资料：《动手学深度学习（PyTorch）》第 2 章
📈 总分：100 分，建议用时 60 分钟

⸻

一、基础理解题（每题 5 分，共 20 分）
	1.	张量（Tensor）是什么？它与 NumPy 的数组有什么异同？
    Tensor是Pytorch用来表示标量的单位，包括向量和矩阵以及多维。
    本质上没有多大区别，只是更简洁且适用于torch。
	2.	什么是自动求导？PyTorch 中如何启用梯度追踪？
    可以自动获取函数的导数，在创建标量时需要标注 requires_grad = True
	3.	requires_grad=True 有什么作用？如果不设置会怎样？
    会对标量自动求导，不设置就无法使用方法 .backward()
	4.	什么情况下需要使用 detach()？它的作用是什么？
    当存在多个变量，且你需要把其中一个变量求导时当作常数。作用是复制一个变量，且求导时视为常数。

二、张量操作题（每题 6 分，共 24 分）
	5.	编写代码创建以下张量，并说明其形状：
    import torch
    A = torch.arange(12).reshape(3, 4)
    3x4的矩阵从0-11顺序。
    6.	对张量 A 进行以下操作，并写出每步后的结果：

	•	取第一行
	•	取每列的最大值
	•	计算列的和（沿着 dim=0）
    print(A[0,:]) -> [0,1,2,3]
    print(A[:,3]) -> [3,7,11]
    print(torch.sum(A,dim=0)) -> [12,15,18,21]

	7.	写出代码实现两个张量相加，其中一个是标量，另一个是二维张量。
        x = torch.tensor(4.0)
        y = torch.tensor((4.0,3.0))
        x + y
	8.	给定张量 A，如何对其进行转置？在 A 是二维张量和三维张量时有何不同？
        A.T
        不同是？


三、自动微分与梯度计算题（每题 8 分，共 24 分）
	9.	写出代码，计算函数 f(x) = 3x^2 + 2x + 1 在 x = 2 处的导数。
        x = torch.tensor(2.0,requires_grad=True)
        f = 3 * x ** 2 + 2 * x + 1
        f.backward()
        print(x.grad)
	10.	写出代码，计算函数 f(x, y) = x * y + y^2 在点 (x=2, y=3) 处的梯度。
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        f = x * y + y ** 2
        f.backward()
        print(x.grad,y.grad)
	11.	使用链式法则，完成如下代码，计算 z 对 x 的梯度：
        x = torch.tensor(2.0, requires_grad=True)
        y = x ** 2
        z = y ** 3
        # TODO: 计算 dz/dx
        y.backward()
        dy_dx = x.grad[0]
        z.backward()
        dz_dy = y.grad[0]
        dz_dx = dz_dy * dy_dx
    
四、数据处理题（每题 8 分，共 16 分）
	12.	用 Pandas 加载一个包含缺失值和类别特征的数据集，完成以下任务：

	•	用均值填充缺失值；
	•	使用 pd.get_dummies() 对类别变量 one-hot 编码；
	•	将其转为 PyTorch 张量。

    ?
	13.	对连续特征进行标准化处理：每一列减去均值再除以标准差。写出完整代码。
    ?

⸻

五、综合编程题（每题 8 分，共 16 分）
	14.	编写一个函数 normalize(X)，将输入的二维张量 X 标准化（均值为 0，方差为 1），返回结果张量。
    def normalize(X):
        return torch.randn(X,0,1)
	15.	写一个完整 PyTorch 自动微分示例：

	•	构造两个变量 x 和 y；
	•	定义函数 f = sin(x) * exp(y)；
	•	计算 f 关于 x 和 y 的梯度。

⸻