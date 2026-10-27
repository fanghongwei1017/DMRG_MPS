# 张量网络作业：DMRG / MPS / Heisenberg 链

对应课件 tensornetwork.pdf 第 73 页的三道作业。报告采用 LaTeX 排版，包含双站点 DMRG 伪代码、能量收敛结果、自旋关联和可点击的仓库链接。

- [PDF 报告](report.pdf)
- [Python 源码](dmrg_heisenberg.py)
- [完整运行数据](results/)
- [LaTeX 源文件](latex/main.tex)

仓库：https://github.com/fanghongwei1017/DMRG_MPS

## 直接查看结果

- `results/energies.csv`：L = 64，各保留态数下的总能量、每格点能量、扫描次数和截断权重。
- `results/energy_convergence.png`：能量随保留态数的变化。
- `results/correlations_center.csv`：固定格点 i = 32，j = 32,...,64 的关联；格点编号从 1 开始。
- `results/spin_correlations.png`：中心自旋关联函数。
- `results/C_dot_matrix.csv`、`C_zz_matrix.csv` 等：完整的 64 × 64 关联矩阵；行列依次对应格点 1,...,64，无表头。
- `results/local_spins.csv`：局域自旋期望值。
- `results/mps_m*.npz`：每个保留态数下的 MPS，各张量依次为 A0,...,A63，指标顺序为（左虚键、物理自旋、右虚键）。
- `results/sweep_history.csv`：所有扫描的收敛记录；一轮扫描包括向右和向左两个方向。
- `results/validation.json`：短链与完整 Hilbert 空间精确对角化的比较。
- `results/metadata.json`：参数、软件版本、数值自检和完整结果。
- `results/correlation_convergence.json`：保存的 m = 64 与 m = 128 波函数之间的关联函数比较。

## 运行

在此文件所在文件夹打开终端。建议 Python 3.10 或更高版本。

```text
python -m pip install -r requirements.txt
python dmrg_heisenberg.py --validate
python dmrg_heisenberg.py --length 64 --bonds 4 8 16 32 64 128
```

默认输出至 `results/`。如需保留本次数据，请指定其他目录：

```text
python dmrg_heisenberg.py --output my_results
```

本次实际环境：Python 3.14.4、NumPy 2.4.4、SciPy 1.18.1；随机种子 7。数值细节可能随 BLAS / 软件版本略有变化。

## 编译报告

安装带 CTeX 与 Fandol 字体的 TeX Live 或 MiKTeX 后，在 `latex/` 目录执行：

```text
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

输出为 `latex/main.pdf`。公式和伪代码均由 LaTeX 排版；两张数值曲线由计算数据绘制，保存在 `latex/images/`。`latex/repository-note.tex` 定义报告中的代码、数据与 TeX 链接。

## 模型与算法

H = Σ(i=1,...,L-1) [Sz_i Sz_(i+1) + (S+_i S-_(i+1) + S-_i S+_(i+1))/2]。

取反铁磁 J = 1、hbar = 1、S = sigma/2，开边界，不加磁场。

`dmrg_heisenberg.py` 自行实现双站点有限 DMRG，使用 NumPy / SciPy，不依赖专用张量网络库。`dmrg()`、`optimize_pair()`、`grow_block()` 是核心算法：

1. 以随机归一化实乘积态初始化 MPS。
2. 保持局部优化区域之外的 MPS 为正交规范，用左右环境构造有效哈密顿量的作用。
3. 使用 `eigsh(which="SA")` 求局部最低本征态；不构造 L = 64 的完整多体矩阵。
4. 对双站点波函数作 SVD，保留最大的 m 个奇异值并重新归一化，移动正交中心。
5. 左右往返扫描，至少扫描 3 轮，连续两次总能量变化小于 1e-9 时停止；最多 40 轮。局部求解容差为 1e-11。
6. 依次增加 m，以上一个 m 的结果作为初态；每轮记录截断后的 MPS 能量。

全程没有量子数标记、固定总 Sz、SU(2) 分块或平移/反射对称性约束。使用实数是因为该哈密顿量和所求基态可以取实数。关联的 x、y、z 分量分别收缩后求和，未强制 C = 3 Czz。

小 m 下，双站点优化之后的截断可能使某轮能量略升；只要求最终扫描满足指定停止标准。扫描收敛不等于严格证明找到了给定 m 的全局最优态。

## 结果

最高保留态数 m = 128 时：

- E = -28.175424859648288。
- E/L = -0.4402410134320045。
- E(64) - E(128) 约为 5.1306e-8。
- Czz(1) = -0.13441476103077876；C(1) = -0.40324428309198657，参考格点为 32。

能量差以 m = 128 的数值结果为参照，不是严格误差上界。开边界使关联依赖两个格点的位置；本报告展示固定参考格点的关联，不假定平移不变性。完整数据在 CSV 中。

短链 L = 2,4,6,8,10 的能量误差不超过 4.45e-15，Czz 的最大误差不超过 3.75e-16。L = 64 的近邻关联之和与独立计算的能量相差约 1.17e-13。

## 参考

- Shoushu Gong，A brief introduction to tensor network states，2026-10-09 课件，第 13、46、73 页。
- U. Schollwoeck, The density-matrix renormalization group in the age of matrix product states, Annals of Physics 326, 96-192 (2011). https://arxiv.org/abs/1008.3477
- SciPy `eigsh` 文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html

本作业按题意使用 AI 辅助编写，所列数值由附带程序实际运行获得。
