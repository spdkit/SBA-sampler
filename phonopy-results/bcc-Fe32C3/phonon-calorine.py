import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, write
from calorine.calculators import CPUNEP
from calorine.tools import relax_structure, get_force_constants
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# =========================================================
# 绘图样式
# =========================================================
plt.rcParams.update({
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
})

# =========================================================
# 参数
# =========================================================
POSCAR         = "POSCAR_1.vasp"
NEP_FILE       = "nep.txt"
SUPERCELL      = [2, 2, 2]
MESH           = [20, 20, 20]
T_MIN, T_MAX, T_STEP = 0, 1000, 100

# k 路径：严格使用 VASPKIT-305 生成的 KPATH.phonopy，不自动生成
KPATH_PHONOPY  = "KPATH.phonopy"
N_PER_SEGMENT  = 100   # 每段插值 k 点数

# =========================================================
# 辅助：LaTeX 标签转换
# =========================================================
LABEL_MAP = {
    "GAMMA":  r"$\Gamma$",
    "DELTA":  r"$\Delta$",
    "SIGMA":  r"$\Sigma$",
    "LAMBDA": r"$\Lambda$",
}

def to_latex(raw: str) -> str:
    return LABEL_MAP.get(raw.upper(), raw)

# =========================================================
# 辅助函数：解析 KPATH.phonopy
# =========================================================
def parse_kpath_phonopy(filename: str, n_per_seg: int = 100):
    """
    解析 VASPKIT-305 生成的 KPATH.phonopy 文件，返回 phonopy 所需的
    bands、ph_labels、path_connections。

    KPATH.phonopy BAND 字段格式：
        BAND = k1x k1y k1z  k2x k2y k2z  ... | k_mx ...  | ...
    其中 '|' 分隔不连续子路径；每个子路径可含任意数量的 k 点（路径途经点）。

    BAND_LABELS 字段格式：
        BAND_LABELS = L1 L2 ... | La Lb ... | ...
    其中 '|' 与 BAND 的子路径对应；每个子路径的标签数必须与 k 点数一致。

    ── Bug 修复说明 ────────────────────────────────────────────────────
    原代码使用正则 r'(?m)^BAND\\s*=\\s*([\\d\\s.\\-eE+]+)' 解析 BAND 字段，
    该字符集不包含 '|'，导致正则在第一个 '|' 处截止，仅捕获首段（2 个
    k 点），与 BAND_LABELS 数量不符，触发 ValueError。

    修复：改用 DOTALL 正则提取整行内容后，再按 '|' 手动分割子路径。
    ────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    filename   : KPATH.phonopy 文件路径
    n_per_seg  : 每段插值 k 点数

    Returns
    -------
    bands            : list[list[list[float]]]  — phonopy bands 参数
    ph_labels        : list[str]                — 高对称点标签（长度 = 段数+1）
    path_connections : list[bool]               — 相邻段是否连通
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"找不到 KPATH.phonopy 文件：{filename}\n"
            f"当前工作目录：{os.getcwd()}\n"
            "请先运行 VASPKIT → 功能 305（Phonopy-format Band-Path）\n"
            "生成 KPATH.phonopy 文件后再重新运行本脚本。"
        )

    with open(filename) as f:
        raw = f.read()

    # 处理行尾续行符（phonopy conf 惯例：行末 '\' 表示续行）
    raw = re.sub(r'\\\s*\n\s*', ' ', raw)

    # ── 提取 BAND 字段 ────────────────────────────────────────────────
    # 匹配 'BAND = ...' 到下一个配置键（大写字母开头的行）或文件末尾
    # 使用 DOTALL 使 '.' 匹配换行，配合懒惰量词 '.*?' 精确截止
    band_m = re.search(
        r'(?m)^BAND\s*=\s*(.*?)(?=\n[ \t]*[A-Z][A-Z0-9_]*[ \t]*[=\n]|\Z)',
        raw, re.DOTALL
    )
    if not band_m:
        raise ValueError(f"未在 {filename} 中找到 BAND 字段。")
    band_raw = band_m.group(1).strip()

    # 按 '|' 分割不连续子路径，每个子路径解析为 k 点列表
    # ── 兼容说明 ───────────────────────────────────────────────────────
    # 不同版本 VASPKIT/phonopy 生成的 KPATH.phonopy 数值格式不同：
    #   空格分隔：  0.000000 0.000000 0.000000
    #   逗号分隔：  0.000000, 0.000000, 0.000000
    #   紧凑逗号：  0.000000,0.000000,0.000000
    # 统一处理：先将逗号/分号替换为空格再 split()，再尝试 float() 转换。
    # ──────────────────────────────────────────────────────────────────
    subpath_kpts = []
    for si, seg_str in enumerate(band_raw.split('|')):
        seg_clean = seg_str.replace(',', ' ').replace(';', ' ')
        tokens = seg_clean.split()
        if not tokens:
            continue
        nums = []
        for t in tokens:
            t_clean = t.strip(',.;:')
            if not t_clean:
                continue
            try:
                nums.append(float(t_clean))
            except ValueError:
                raise ValueError(
                    f"BAND 字段第 {si+1} 子路径含无法转换的 token：{t!r}\n"
                    f"（strip 后为 {t_clean!r}）\n"
                    f"原始内容：{seg_str!r}"
                )
        if len(nums) % 3 != 0:
            raise ValueError(
                f"BAND 第 {si+1} 子路径数值个数 ({len(nums)}) 不是 3 的倍数。\n"
                f"原始内容：{seg_str!r}"
            )
        kpts = [nums[i:i+3] for i in range(0, len(nums), 3)]
        if len(kpts) < 2:
            raise ValueError(
                f"BAND 第 {si+1} 子路径只有 {len(kpts)} 个 k 点，至少需要 2 个。"
            )
        subpath_kpts.append(kpts)

    if not subpath_kpts:
        raise ValueError(f"BAND 字段中未解析到任何有效子路径。")

    # ── 提取 BAND_LABELS 字段 ─────────────────────────────────────────
    labels_m = re.search(
        r'(?m)^BAND_LABELS\s*=\s*(.*?)(?=\n[ \t]*[A-Z][A-Z0-9_]*[ \t]*[=\n]|\Z)',
        raw, re.DOTALL
    )
    label_subs = None
    if labels_m:
        labels_raw = labels_m.group(1).strip()
        label_subs = [
            [lb for lb in s.split() if lb.strip()]
            for s in labels_raw.split('|')
        ]
        # 校验：各子路径标签数必须与 k 点数一致
        if len(label_subs) != len(subpath_kpts):
            raise ValueError(
                f"BAND_LABELS 子路径数 ({len(label_subs)}) "
                f"与 BAND 子路径数 ({len(subpath_kpts)}) 不符。\n"
                f"请检查 KPATH.phonopy 中 '|' 的数量是否一致。"
            )
        for si, (kpts, lbls) in enumerate(zip(subpath_kpts, label_subs)):
            if len(lbls) != len(kpts):
                raise ValueError(
                    f"BAND_LABELS 第 {si+1} 子路径标签数 ({len(lbls)}) "
                    f"与该子路径 k 点数 ({len(kpts)}) 不符。\n"
                    f"标签：{lbls}"
                )

    # ── 构建 phonopy bands 和 path_connections ────────────────────────
    # 每对相邻途经点之间均匀插值 n_per_seg 个 k 点形成一段（segment）。
    # path_connections[i] = True  → 第 i 段与第 i+1 段连续
    # path_connections[i] = False → 第 i 段是某子路径的最后一段
    bands            = []
    path_connections = []

    for si, kpts in enumerate(subpath_kpts):
        n_wp = len(kpts)
        for ki in range(n_wp - 1):
            k1  = np.array(kpts[ki])
            k2  = np.array(kpts[ki + 1])
            seg = [(k1 + t * (k2 - k1)).tolist()
                   for t in np.linspace(0.0, 1.0, n_per_seg)]
            bands.append(seg)

            if ki < n_wp - 2:
                # 子路径内部：相邻段连续
                path_connections.append(True)
            elif si < len(subpath_kpts) - 1:
                # 子路径末段，后续还有子路径：不连续
                path_connections.append(False)
            else:
                # 整条路径的最后一段
                path_connections.append(False)

    # ── 构建 ph_labels（长度 = len(bands) + 1）───────────────────────
    # phonopy 约定：断开处将两端标签合并为 "A|B"
    if label_subs is not None:
        ph_labels = []
        for si, (kpts, lbls) in enumerate(zip(subpath_kpts, label_subs)):
            for ki, lbl in enumerate(lbls):
                if si == 0 and ki == 0:
                    ph_labels.append(lbl)
                elif ki == 0:
                    # 新子路径起点：与上一个标签合并为 "prevEnd|newStart"
                    ph_labels[-1] = ph_labels[-1] + '|' + lbl
                else:
                    ph_labels.append(lbl)
    else:
        # 无标签：使用数字占位
        ph_labels = [str(i) for i in range(len(bands) + 1)]

    n_segs   = len(bands)
    n_labels = len(ph_labels)
    if n_labels != n_segs + 1:
        raise ValueError(
            f"构建标签数 ({n_labels}) 与段数+1 ({n_segs+1}) 不符，"
            "请检查 KPATH.phonopy 格式。"
        )

    return bands, ph_labels, path_connections


# =========================================================
# 辅助函数 C：导出 band.dat
# =========================================================
def export_band_dat(phonon, filename="band.dat"):
    band_dict   = phonon.get_band_structure_dict()
    distances   = band_dict["distances"]
    frequencies = band_dict["frequencies"]

    with open(filename, "w") as f:
        for i, dist in enumerate(distances):
            for j in range(len(dist)):
                line = f"{dist[j]:.8f}"
                for k in range(len(frequencies[i][j])):
                    line += f" {frequencies[i][j][k]:.8f}"
                f.write(line + "\n")
            f.write("\n")

    print(f"{filename} saved.\n")


# =========================================================
# 辅助函数 D：导出 dos.dat
# =========================================================
def export_dos_dat(phonon, filename="dos.dat"):
    dos_dict = phonon.get_total_dos_dict()
    freq     = dos_dict["frequency_points"]
    dos      = dos_dict["total_dos"]

    with open(filename, "w") as f:
        for i in range(len(freq)):
            f.write(f"{freq[i]:.8f} {dos[i]:.8f}\n")

    print(f"{filename} saved.\n")


# =========================================================
# 1  读入结构
# =========================================================
print("\nReading POSCAR...")
atoms = read(POSCAR)

# =========================================================
# 2  加载 NEP 势函数
# =========================================================
print("Loading NEP potential...")
calculator = CPUNEP(NEP_FILE)
atoms.calc = calculator

# =========================================================
# 3  结构弛豫
# =========================================================
print("Relaxing structure...")
relax_structure(atoms, fmax=1e-2)
write("relaxed_POSCAR.vasp", atoms)
print("Relaxation finished.\n")

# =========================================================
# 4  计算力常数
# =========================================================
print("Calculating force constants...")
phonon_fc       = get_force_constants(atoms, calculator, SUPERCELL)
force_constants = phonon_fc.force_constants
print("Force constants done.\n")

# =========================================================
# 5  构建 Phonopy 对象
# =========================================================
phonopy_atoms = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell[:],
    scaled_positions=atoms.get_scaled_positions(),
)
phonon = Phonopy(phonopy_atoms, supercell_matrix=np.diag(SUPERCELL))
phonon.force_constants = force_constants

# =========================================================
# 6  高对称 k 路径
#    严格使用 KPATH.phonopy（VASPKIT-305 输出），不使用 seekpath 自动生成。
#    若文件缺失或格式有误，直接报错，不回退到其他来源。
# =========================================================
print("Generating high-symmetry k-path...")
print(f"  读取 {KPATH_PHONOPY} ...")

bands, ph_labels, path_connections = parse_kpath_phonopy(
    KPATH_PHONOPY, n_per_seg=N_PER_SEGMENT
)

print(f"  路径段数: {len(bands)}, "
      f"path_connections ({len(path_connections)}): {path_connections}")
print(f"  Labels ({len(ph_labels)}): {ph_labels}\n")

# =========================================================
# 7  声子能带结构
# =========================================================
print("Calculating phonon band structure...")
phonon.run_band_structure(
    bands,
    labels=ph_labels,
    path_connections=path_connections,
)

phonon.plot_band_structure()
fig_band = plt.gcf()
fig_band.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.10)
fig_band.savefig("phonon_band.png", dpi=400)
plt.close(fig_band)
print("phonon_band.png saved.\n")

export_band_dat(phonon, "band.dat")

# =========================================================
# 8  声子态密度
# =========================================================
print("Calculating phonon DOS...")
phonon.run_mesh(MESH)
phonon.run_total_dos()

phonon.plot_total_dos()
fig_dos = plt.gcf()
ax_dos  = fig_dos.get_axes()[0]
ax_dos.set_xlabel("Frequency (THz)")
ax_dos.set_ylabel("DOS (states/THz)")
fig_dos.tight_layout()
fig_dos.savefig("phonon_dos.png", dpi=400)
plt.close(fig_dos)
print("phonon_dos.png saved.\n")

export_dos_dat(phonon, "dos.dat")

# =========================================================
# 9  热力学性质
# =========================================================
print("Calculating thermal properties...")
phonon.run_thermal_properties(t_min=T_MIN, t_max=T_MAX, t_step=T_STEP)
tp = phonon.get_thermal_properties_dict()
T  = tp["temperatures"]
F  = tp["free_energy"]
S  = tp["entropy"]
Cv = tp["heat_capacity"]

# =========================================================
# 10  热力学性质图
# =========================================================
print("Plotting thermal properties...")

fig_cv, ax_cv = plt.subplots()
ax_cv.plot(T, Cv, linewidth=2)
ax_cv.set_xlabel("Temperature (K)")
ax_cv.set_ylabel(r"$C_V$  (J K$^{-1}$ mol$^{-1}$)")
fig_cv.tight_layout()
fig_cv.savefig("Cv_T.png", dpi=400)
plt.close(fig_cv)

fig_f, ax_f = plt.subplots()
ax_f.plot(T, F, linewidth=2)
ax_f.set_xlabel("Temperature (K)")
ax_f.set_ylabel(r"Free energy (kJ mol$^{-1}$)")
fig_f.tight_layout()
fig_f.savefig("F_T.png", dpi=400)
plt.close(fig_f)

print("Thermal plots saved.\n")

# =========================================================
# 11  导出 thermal_properties.txt
# =========================================================
print("Writing thermal_properties.txt...")

natom                = len(atoms)
num_modes            = natom * 3
num_integrated_modes = num_modes - 3
zpe                  = F[0]

with open("thermal_properties.txt", "w") as f:
    f.write("Thermal properties / unit cell\n\n")
    f.write("unit:\n")
    f.write("  temperature:   K\n")
    f.write("  free_energy:   kJ/mol\n")
    f.write("  entropy:       J/K/mol\n")
    f.write("  heat_capacity: J/K/mol\n\n")
    f.write(f"natom: {natom}\n")
    f.write("cutoff_frequency: 0.00000\n")
    f.write(f"num_modes: {num_modes}\n")
    f.write(f"num_integrated_modes: {num_integrated_modes}\n\n")
    f.write(f"zero_point_energy: {zpe:.7f}\n\n")
    f.write("thermal_properties:\n")
    for i in range(len(T)):
        energy = F[i] + T[i] * S[i] / 1000.0
        f.write(f"- temperature:   {T[i]:12.7f}\n")
        f.write(f"  free_energy:   {F[i]:12.7f}\n")
        f.write(f"  entropy:       {S[i]:12.7f}\n")
        f.write(f"  heat_capacity: {Cv[i]:12.7f}\n")
        f.write(f"  energy:        {energy:12.7f}\n")

print("thermal_properties.txt saved.\n")

# =========================================================
# 12  保存 phonopy.yaml
# =========================================================
print("Saving phonopy.yaml...")
phonon.save("phonopy.yaml")
print("phonopy.yaml saved.\n")

# =========================================================
# 完成
# =========================================================
print("=" * 42)
print("   NEP phonon workflow finished")
print("=" * 42)
print("Generated files:")
for fname in [
    "relaxed_POSCAR.vasp", "phonon_band.png", "phonon_dos.png",
    "band.dat", "dos.dat",
    "Cv_T.png", "F_T.png", "thermal_properties.txt", "phonopy.yaml",
]:
    print(f"  {fname}")
