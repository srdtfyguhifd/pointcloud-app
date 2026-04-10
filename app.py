"""
文件名：streamlit_app.py
功能：将完整的点云预处理Pipeline封装为Web应用（含可视化）
"""
import streamlit as st
import open3d as o3d
import numpy as np
import tempfile
import os
import plotly.graph_objects as go
from datetime import datetime

# ==================== 默认配置 ====================
DEFAULT_CONFIG = {
    'distance_filter': {
        'min_distance': 0.8,
        'max_distance': 2.0,
        'enabled': True
    },
    'statistical_filter': {
        'nb_neighbors': 70,
        'std_ratio': 4.0,
        'enabled': True
    },
    'voxel_downsample': {
        'voxel_size': 0.01,
        'enabled': True
    },
    'ground_segmentation': {
        'distance_threshold': 0.005,
        'num_iterations': 1000,
        'enabled': True
    }
}

# ==================== 预处理函数 ====================
def distance_filter(pcd, min_distance=0.8, max_distance=2.0, verbose=True):
    """
    距离滤波：基于预设距离范围过滤点云

    参数：
    ----------
    pcd : open3d.geometry.PointCloud
        输入点云
    min_distance : float
        最小有效距离 (米)
    max_distance : float
        最大有效距离 (米)
    verbose : bool
        是否打印详细信息

    返回：
    -------
    filtered_pcd : open3d.geometry.PointCloud
        滤波后的点云
    """
    points = np.asarray(pcd.points)

    # 计算每个点到原点的距离
    distances = np.linalg.norm(points, axis=1)

    # 根据距离范围筛选
    mask = (distances >= min_distance) & (distances <= max_distance)
    filtered_points = points[mask]

    # 创建新的点云
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 如果原始点云有颜色，保留颜色
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    if verbose:
        removed = len(points) - len(filtered_points)
        print(f"  原始点数: {len(points)}")
        print(f"  移除点数: {removed} ({removed/len(points)*100:.1f}%)")
        print(f"  剩余点数: {len(filtered_points)}")
        print(f"  距离范围: [{min_distance:.2f}, {max_distance:.2f}] 米")

    return filtered_pcd


def statistical_filter(pcd, nb_neighbors=70, std_ratio=4.0, verbose=True):
    """
    统计滤波：移除离群点

    参数：
    ----------
    pcd : open3d.geometry.PointCloud
        输入点云
    nb_neighbors : int
        邻居点数量
    std_ratio : float
        标准差倍数阈值
    verbose : bool
        是否打印详细信息

    返回：
    -------
    filtered_pcd : open3d.geometry.PointCloud
        滤波后的点云
    """
    if len(pcd.points) == 0:
        return pcd

    # Open3D 内置统计滤波
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )

    if verbose:
        removed = len(pcd.points) - len(ind)
        print(f"  原始点数: {len(pcd.points)}")
        print(f"  移除离群点: {removed} ({removed/len(pcd.points)*100:.1f}%)")
        print(f"  剩余点数: {len(ind)}")
        print(f"  参数: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")

    return cl


def voxel_downsample(pcd, voxel_size=0.01, verbose=True):
    """
    体素下采样：减少点云密度

    参数：
    ----------
    pcd : open3d.geometry.PointCloud
        输入点云
    voxel_size : float
        体素大小 (米)
    verbose : bool
        是否打印详细信息

    返回：
    -------
    downsampled_pcd : open3d.geometry.PointCloud
        下采样后的点云
    """
    if len(pcd.points) == 0:
        return pcd

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    if verbose:
        removed = len(pcd.points) - len(downsampled_pcd.points)
        print(f"  原始点数: {len(pcd.points)}")
        print(f"  下采样后点数: {len(downsampled_pcd.points)}")
        print(f"  减少点数: {removed} ({removed/len(pcd.points)*100:.1f}%)")
        print(f"  体素大小: {voxel_size:.3f} 米")

    return downsampled_pcd


def segment_ground(pcd, distance_threshold=0.005, num_iterations=1000, verbose=True):
    """
    地面分割：使用RANSAC平面拟合分割地面

    参数：
    ----------
    pcd : open3d.geometry.PointCloud
        输入点云
    distance_threshold : float
        RANSAC距离阈值
    num_iterations : int
        RANSAC迭代次数
    verbose : bool
        是否打印详细信息

    返回：
    -------
    ground_pcd : open3d.geometry.PointCloud
        地面点云
    objects_pcd : open3d.geometry.PointCloud
        非地面点云（物体）
    plane_model : list
        平面模型参数 [a, b, c, d]
    """
    if len(pcd.points) < 3:
        return pcd, o3d.geometry.PointCloud(), None

    # RANSAC 平面分割
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations
    )

    # 分割地面和非地面
    ground_pcd = pcd.select_by_index(inliers)
    objects_pcd = pcd.select_by_index(inliers, invert=True)

    if verbose:
        print(f"  平面模型: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
        print(f"  地面点数: {len(ground_pcd.points)}")
        print(f"  物体点数: {len(objects_pcd.points)}")
        print(f"  地面占比: {len(ground_pcd.points)/len(pcd.points)*100:.1f}%")
        print(f"  参数: distance_threshold={distance_threshold}, num_iterations={num_iterations}")

    return ground_pcd, objects_pcd, plane_model


def preprocess_single(pcd, config=None, verbose=True):
    """
    对单个点云执行完整预处理Pipeline

    参数说明：
    ----------
    pcd : open3d.geometry.PointCloud
        输入点云
    config : dict
        配置字典，None则使用默认配置
    verbose : bool
        是否打印处理信息

    返回值：
    -------
    processed_pcd : open3d.geometry.PointCloud
        预处理后的点云（去除地面后的物体）
    stats : dict
        处理统计信息
    """

    if config is None:
        config = DEFAULT_CONFIG

    # 记录统计信息
    stats = {
        'original_points': len(pcd.points),
        'steps': []
    }

    current_pcd = pcd

    # ===== Step 1: 距离滤波 =====
    if config['distance_filter']['enabled']:
        if verbose:
            print("\n[Step 1/4] 距离滤波")

        points_before = len(current_pcd.points)
        current_pcd = distance_filter(
            current_pcd,
            min_distance=config['distance_filter']['min_distance'],
            max_distance=config['distance_filter']['max_distance'],
            verbose=verbose
        )
        stats['steps'].append({
            'name': '距离滤波',
            'points_before': points_before,
            'points_after': len(current_pcd.points),
            'removed': points_before - len(current_pcd.points)
        })

    # ===== Step 2: 统计滤波 =====
    if config['statistical_filter']['enabled']:
        if verbose:
            print("\n[Step 2/4] 统计滤波")

        points_before = len(current_pcd.points)
        current_pcd = statistical_filter(
            current_pcd,
            nb_neighbors=config['statistical_filter']['nb_neighbors'],
            std_ratio=config['statistical_filter']['std_ratio'],
            verbose=verbose
        )
        stats['steps'].append({
            'name': '统计滤波',
            'points_before': points_before,
            'points_after': len(current_pcd.points),
            'removed': points_before - len(current_pcd.points)
        })

    # ===== Step 3: 体素下采样 =====
    if config['voxel_downsample']['enabled']:
        if verbose:
            print("\n[Step 3/4] 体素下采样")

        points_before = len(current_pcd.points)
        current_pcd = voxel_downsample(
            current_pcd,
            voxel_size=config['voxel_downsample']['voxel_size'],
            verbose=verbose
        )
        stats['steps'].append({
            'name': '体素下采样',
            'points_before': points_before,
            'points_after': len(current_pcd.points),
            'removed': points_before - len(current_pcd.points)
        })

    # ===== Step 4: 地面分割 =====
    if config['ground_segmentation']['enabled']:
        if verbose:
            print("\n[Step 4/4] 地面分割")

        points_before = len(current_pcd.points)
        ground_pcd, objects_pcd, plane_model = segment_ground(
            current_pcd,
            distance_threshold=config['ground_segmentation']['distance_threshold'],
            num_iterations=config['ground_segmentation']['num_iterations'],
            verbose=verbose
        )

        current_pcd = objects_pcd  # 我们要的是非地面部分

        stats['steps'].append({
            'name': '地面分割',
            'points_before': points_before,
            'ground_points': len(ground_pcd.points),
            'object_points': len(objects_pcd.points),
            'plane_model': plane_model
        })

    stats['final_points'] = len(current_pcd.points)
    stats['reduction_ratio'] = 1 - stats['final_points'] / stats['original_points']

    return current_pcd, stats


# ==================== 可视化函数 ====================
def create_point_cloud_plot(pcd, max_points=50000, color_by='height'):
    """
    将Open3D点云转换为Plotly可视化对象

    Args:
        pcd: Open3D点云对象
        max_points: 最大显示点数（避免浏览器卡顿）
        color_by: 着色方式 ('height' 或 'uniform')
    """
    if len(pcd.points) == 0:
        return go.Figure()

    # 转换为numpy数组
    points = np.asarray(pcd.points)

    # 如果点太多，进行随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[indices]
        else:
            colors = None
    else:
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # 创建3D散点图
    if colors is not None and color_by == 'original':
        # 使用原始颜色
        marker_color = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                       for r, g, b in colors]
    else:
        # 根据高度着色
        z_normalized = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-10)
        marker_color = z_normalized

    # 创建Plotly图形
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2.0,
            color=marker_color,
            colorscale='Viridis',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="高度 (m)")
        ),
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    )])

    # 设置布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False,
        hovermode='closest'
    )

    return fig


def get_point_cloud_stats(pcd):
    """获取点云统计信息"""
    if len(pcd.points) == 0:
        return {'点数': 0, 'X范围': 'N/A', 'Y范围': 'N/A', 'Z范围': 'N/A'}

    points = np.asarray(pcd.points)
    return {
        '点数': len(points),
        'X范围': f"[{points[:,0].min():.3f}, {points[:,0].max():.3f}]",
        'Y范围': f"[{points[:,1].min():.3f}, {points[:,1].max():.3f}]",
        'Z范围': f"[{points[:,2].min():.3f}, {points[:,2].max():.3f}]",
        '体积 (m³)': f"{pcd.get_axis_aligned_bounding_box().volume():.3f}"
    }


# ==================== Web 界面 ====================
st.set_page_config(
    page_title="点云预处理服务",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 点云预处理在线工具")
st.markdown("上传你的 `.pcd` 或 `.ply` 文件，执行完整的预处理流程（距离滤波、统计滤波、体素下采样、地面分割）")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 预处理参数配置")

    st.subheader("1️⃣ 距离滤波")
    enable_distance = st.checkbox("启用距离滤波", value=DEFAULT_CONFIG['distance_filter']['enabled'])
    min_distance = st.number_input(
        "最小距离 (m)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_CONFIG['distance_filter']['min_distance'],
        step=0.1,
        format="%.2f"
    )
    max_distance = st.number_input(
        "最大距离 (m)",
        min_value=0.0,
        max_value=10.0,
        value=DEFAULT_CONFIG['distance_filter']['max_distance'],
        step=0.1,
        format="%.2f"
    )

    st.subheader("2️⃣ 统计滤波")
    enable_statistical = st.checkbox("启用统计滤波", value=DEFAULT_CONFIG['statistical_filter']['enabled'])
    nb_neighbors = st.number_input(
        "邻居点数量",
        min_value=10,
        max_value=200,
        value=DEFAULT_CONFIG['statistical_filter']['nb_neighbors'],
        step=10
    )
    std_ratio = st.number_input(
        "标准差倍数",
        min_value=1.0,
        max_value=10.0,
        value=DEFAULT_CONFIG['statistical_filter']['std_ratio'],
        step=0.5,
        format="%.1f"
    )

    st.subheader("3️⃣ 体素下采样")
    enable_voxel = st.checkbox("启用体素下采样", value=DEFAULT_CONFIG['voxel_downsample']['enabled'])
    voxel_size = st.number_input(
        "体素大小 (m)",
        min_value=0.001,
        max_value=0.1,
        value=DEFAULT_CONFIG['voxel_downsample']['voxel_size'],
        step=0.001,
        format="%.3f"
    )

    st.subheader("4️⃣ 地面分割")
    enable_ground = st.checkbox("启用地面分割", value=DEFAULT_CONFIG['ground_segmentation']['enabled'])
    distance_threshold = st.number_input(
        "RANSAC距离阈值 (m)",
        min_value=0.001,
        max_value=0.1,
        value=DEFAULT_CONFIG['ground_segmentation']['distance_threshold'],
        step=0.001,
        format="%.3f"
    )
    num_iterations = st.number_input(
        "RANSAC迭代次数",
        min_value=100,
        max_value=5000,
        value=DEFAULT_CONFIG['ground_segmentation']['num_iterations'],
        step=100
    )

    st.markdown("---")
    st.subheader("🎨 可视化设置")
    max_display_points = st.slider(
        "最大显示点数",
        min_value=10000,
        max_value=200000,
        value=50000,
        step=10000,
        help="限制显示的点数以避免浏览器卡顿"
    )

    color_mode = st.radio(
        "着色方式",
        options=['height', 'original'],
        format_func=lambda x: "按高度着色" if x == 'height' else "原始颜色",
        help="选择点云的着色方式"
    )

# 主界面
uploaded_file = st.file_uploader(
    "📂 选择点云文件",
    type=['pcd', 'ply', 'xyz', 'xyzn', 'xyzrgb', 'pts'],
    help="支持 PCD, PLY, XYZ 等格式"
)

if uploaded_file is not None:
    # 显示文件信息
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.info(f"📄 文件名: {uploaded_file.name} | 大小: {file_size:.2f} MB")

    # 构建配置
    config = {
        'distance_filter': {
            'min_distance': min_distance,
            'max_distance': max_distance,
            'enabled': enable_distance
        },
        'statistical_filter': {
            'nb_neighbors': nb_neighbors,
            'std_ratio': std_ratio,
            'enabled': enable_statistical
        },
        'voxel_downsample': {
            'voxel_size': voxel_size,
            'enabled': enable_voxel
        },
        'ground_segmentation': {
            'distance_threshold': distance_threshold,
            'num_iterations': num_iterations,
            'enabled': enable_ground
        }
    }

    # --- 处理逻辑 ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_input_path = tmp_file.name

    try:
        # 使用 Open3D 读取
        pcd = o3d.io.read_point_cloud(temp_input_path)

        if len(pcd.points) == 0:
            st.error("❌ 无法读取点云数据，请检查文件格式。")
        else:
            # 调用预处理函数
            with st.spinner('🔄 正在处理点云... (这可能需要几秒钟)'):
                processed_pcd, stats = preprocess_single(pcd, config, verbose=True)

            # --- 前端可视化展示 ---
            st.markdown("---")
            st.subheader("📊 点云对比可视化")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 原始点云")
                stats_original = get_point_cloud_stats(pcd)
                st.json(stats_original, expanded=False)

                with st.spinner('正在渲染原始点云...'):
                    fig_original = create_point_cloud_plot(
                        pcd,
                        max_points=max_display_points,
                        color_by=color_mode
                    )
                    st.plotly_chart(fig_original, use_container_width=True)

            with col2:
                st.markdown("### 处理后点云")
                stats_processed = get_point_cloud_stats(processed_pcd)
                st.json(stats_processed, expanded=False)

                with st.spinner('正在渲染处理后点云...'):
                    fig_processed = create_point_cloud_plot(
                        processed_pcd,
                        max_points=max_display_points,
                        color_by=color_mode
                    )
                    st.plotly_chart(fig_processed, use_container_width=True)

            # 显示处理步骤详情
            st.markdown("---")
            st.subheader("📈 处理步骤详情")

            # 创建处理步骤表格
            steps_data = []
            for step in stats['steps']:
                if step['name'] == '地面分割':
                    steps_data.append({
                        '步骤': step['name'],
                        '处理前点数': step['points_before'],
                        '处理后点数': step['object_points'],
                        '移除点数': step['points_before'] - step['object_points'],
                        '移除比例': f"{(step['points_before'] - step['object_points']) / step['points_before'] * 100:.1f}%"
                    })
                else:
                    steps_data.append({
                        '步骤': step['name'],
                        '处理前点数': step['points_before'],
                        '处理后点数': step['points_after'],
                        '移除点数': step['removed'],
                        '移除比例': f"{step['removed'] / step['points_before'] * 100:.1f}%"
                    })

            # 显示表格
            if steps_data:
                import pandas as pd
                df = pd.DataFrame(steps_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # 总体统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "原始点数",
                    f"{stats['original_points']:,}",
                )
            with col2:
                st.metric(
                    "最终点数",
                    f"{stats['final_points']:,}",
                )
            with col3:
                reduction_ratio = stats['reduction_ratio'] * 100
                st.metric(
                    "减少比例",
                    f"{reduction_ratio:.1f}%",
                    delta=f"{stats['original_points'] - stats['final_points']:,} 点"
                )
            with col4:
                processing_time = datetime.now().strftime("%H:%M:%S")
                st.metric(
                    "处理时间",
                    processing_time
                )

            # --- 文件下载 ---
            st.markdown("---")
            st.subheader("💾 导出处理结果")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pcd") as out_tmp:
                o3d.io.write_point_cloud(out_tmp.name, processed_pcd)
                with open(out_tmp.name, "rb") as f:
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        st.download_button(
                            label="⬇️ 下载处理后的点云 (PCD格式)",
                            data=f.read(),
                            file_name=f"processed_{uploaded_file.name.split('.')[0]}.pcd",
                            mime="application/octet-stream",
                            use_container_width=True
                        )

            os.unlink(out_tmp.name)

    except Exception as e:
        st.error(f"❌ 处理出错: {str(e)}")
        st.exception(e)
    finally:
        os.unlink(temp_input_path)

else:
    # 显示欢迎信息
    st.markdown("---")
    st.markdown("""
    ### 📖 使用说明
    
    **完整预处理流程：**
    1. **距离滤波**：基于距离范围过滤点云，移除过近或过远的点
    2. **统计滤波**：移除离群噪点，提高点云质量
    3. **体素下采样**：降低点云密度，加快处理速度
    4. **地面分割**：使用RANSAC算法识别并移除地面点云
    
    **操作步骤：**
    1. 📂 上传点云文件（支持 PCD、PLY、XYZ 等格式）
    2. ⚙️ 在侧边栏调整预处理参数（或使用默认值）
    3. 🔄 系统自动执行完整的预处理流程
    4. 📊 查看原始点云和处理后点云的对比
    5. 💾 下载处理后的点云文件
    
    ### 💡 参数说明
    - **距离滤波**：适合移除传感器近场噪点和远场稀疏点
    - **统计滤波**：移除明显偏离主体的离群点
    - **体素下采样**：降低点云密度，建议体素大小 0.01-0.02m
    - **地面分割**：RANSAC 平面拟合，阈值越小越严格
    
    ### 🎯 默认参数（优化配置）
    - 距离范围：0.8-2.0m（适合桌面场景）
    - 统计滤波：70邻居，4.0标准差
    - 体素大小：0.01m（保持精度）
    - RANSAC阈值：0.005m（精确地面分割）
    """)

# 页脚
st.markdown("---")
st.markdown("© 2024 点云预处理服务 | Powered by Open3D & Streamlit | 完整Pipeline实现")