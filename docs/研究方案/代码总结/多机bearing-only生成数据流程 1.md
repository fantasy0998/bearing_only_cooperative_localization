sync_three_robot_states
得到共同的时间戳以及三机在全局坐标系下的位姿

将start_point点的位姿作为姿态冻结窗口起始点，即B2坐标系，同样对于新的无人机C而言，也有C2坐标系

载入一个图，行时观测对象，列是观测者，存在这样的观测就将其元素置为1，否则为0

对于值为1的观测边，调用synthesize_directed_bearing_edge生成该边的观测

返回的观测量中只包含experiment_dir，anchor_id，robot_ids，timestamps，states(包含世界坐标和世界坐标与局部坐标的旋转和平移，参考离散采样点生成的代码中有没有在旋转平移后的点上加噪声)，observations
（包含所有边的观测）


`````` python
config = MiluvBearingExperimentConfig(
            experiment_dir=MILUV_EXPERIMENT_DIR,
            output_root=output_root,
            window_sizes=(20,),
            sigma_azimuth_deg=(0.1,),
            sigma_elevation_deg=(0.5,),
            window_stride=10,
            keypoint_counts=(6, 12, 18),
            dt=0.05,
            random_seed=7,
            force_recompute=True,
            max_samples=800,  # 原始数据中一共有8100历元，时间间隔0.01s左右，生成的观测量需要对齐时间戳，按照dt采样时间生成时间序列，设置采样40s数据，即800个点
            max_windows=1,
            start_point=2000,
            end_point=6000,
        )
``````



增加了def build_observation_matrix(anchor_nodes, estimated_nodes)函数
通过图的形式生成候选边集合
因此后续生成的观测量数量全部依赖这个函数生成的adjcency matrix进行计算，结果存到一个字典中，不再有B_xx，C_xx这种单独的命名变量，节点命名来自于anchor_nodes和estimated_nodes, 全部统一为“ifo1”，"ifo2"...,观测量，轨迹，姿态等变量保存形式参考下面的字典结构
"estiamated_pose":{"ifo2":#pose of ifo2 at start point, 
						   "ifo3":#pose of ifo2 at start point}
这样方便后续增加更多无人机

下面的代码不考虑window_size， 只考虑噪声大小即可，即组合不同的噪声，如输入(0.1 0.2 0.3)(0.4,0.5,0.6)
组合成(0.1,0.4)(0.2,0.5)(0.3,0.6)
````python
    def specs(self) -> list[ObservationBatchSpec]:
        """Enumerate all observation-batch combinations."""
        specs: list[ObservationBatchSpec] = [] # 类型注解 定义一下specs的类型为列表，列表中的元素类型为ObservationBatchSpec
        for window_size in self.config.window_sizes:
            for sigma_az, sigma_el in zip(
                self.config.sigma_azimuth_deg, self.config.sigma_elevation_deg
            ):
                specs.append(
                    ObservationBatchSpec(
                        window_size=window_size,
                        sigma_azimuth_deg=float(sigma_az),
                        sigma_elevation_deg=float(sigma_el),
                    )
                )
        return specs
````


这里的directed_edges由上面的邻接矩阵中存在观测的边生成
```` python
for observer_id, target_id in directed_edges:
        if observer_id not in states or target_id not in states:
            raise ValueError(
                f"Edge ({observer_id}, {target_id}) references unknown robots."
            )
  
        edge_key = f"{observer_id}<-{target_id}"
        edge_observation = synthesize_directed_bearing_edge(
            observer_position_w=states[observer_id]["position_w"],
            observer_rotation_wb=states[observer_id]["rotation_wb"],
            target_position_w=states[target_id]["position_w"],
            reference_index=reference_index,  ## 参考坐标系
            sigma_azimuth_deg=sigma_azimuth_deg,
            sigma_elevation_deg=sigma_elevation_deg,
            rng=generator,
        )
        observations[edge_key] = {
            "observer_id": observer_id,
            "target_id": target_id,
            **edge_observation,
        }
````

synthesize_three_robot_bearing_observations函数返回的字典需要包含这段轨迹起始时刻的位姿，没必要包含reference_index, 每一段轨迹的reference pose都是开头时刻的姿态，所以改为下面的形式
````python
    return {
        "experiment_dir": str(experiment_dir),
        "timeshift_s": sync_data["timeshift_s"],
        "anchor_id": anchor_id,
        "robot_ids": tuple(robot_ids),
        "estiamated_pose":{"ifo2":#pose of ifo2 at start point, 
						   "ifo3":#pose of ifo2 at start point}
        "timestamps": timestamps,
        "states": states,
        "observations": observations,
    }

````


在后续代码的命名中不使用ABC代指无人机
````` python
np.savez_compressed(
            output_path,
            experiment_dir=str(self.config.experiment_dir),
            window_size=np.array(spec.window_size, dtype=int),
            sigma_azimuth_deg=np.array(spec.sigma_azimuth_deg, dtype=float),
            sigma_elevation_deg=np.array(spec.sigma_elevation_deg, dtype=float),
            timestamps=dataset["timestamps"],
            anchor_position_w=states["ifo001"]["position_w"],
            B_position_w=states["ifo002"]["position_w"],
            C_position_w=states["ifo003"]["position_w"],
            B_rotation_wb=states["ifo002"]["rotation_wb"],
            C_rotation_wb=states["ifo003"]["rotation_wb"],
            B_from_A_bearing_b4=observations["ifo002<-ifo001"]["bearing_b4"],
            C_from_A_bearing_b4=observations["ifo003<-ifo001"]["bearing_b4"],
            B_from_C_bearing_b4=observations["ifo002<-ifo003"]["bearing_b4"],
            C_from_B_bearing_b4=observations["ifo003<-ifo002"]["bearing_b4"],
        )
`````
上面的保存结果部分代码修改为下面的形式：每一组噪声保存两组数据，一组为观测量，一组为解算结果的参照
纯观测数据包含
````python
            experiment_dir=str(self.config.experiment_dir),
            window_size=np.array(spec.window_size, dtype=int),
            sigma_azimuth_deg=np.array(spec.sigma_azimuth_deg, dtype=float),
            sigma_elevation_deg=np.array(spec.sigma_elevation_deg, dtype=float),
            timestamps=dataset["timestamps"],
            anchor_position_w=states["ifo001"]["position_w"],
          
            estiamated_position_l={"ifo2":states["ifo002"]["position_l"], 
						           "ifo3":states["ifo003"]["position_l"], }
            
			bearing_observation={"ifo2_ifo1":observations["ifo002<-ifo001"]["bearing_b4"]
								"ifo3_ifo1":observations["ifo003<-ifo001"]["bearing_b4"]
								"ifo2_ifo3":observations["ifo002<-ifo003"]["bearing_b4"]
								"ifo3_ifo2":observations["ifo003<-ifo002"]["bearing_b4"]

````
结算结果的参照数据包含
````python
            experiment_dir=str(self.config.experiment_dir),
            window_size=np.array(spec.window_size, dtype=int),
            sigma_azimuth_deg=np.array(spec.sigma_azimuth_deg, dtype=float),
            sigma_elevation_deg=np.array(spec.sigma_elevation_deg, dtype=float),
            timestamps=dataset["timestamps"],

            estiamated_position_w={"ifo2":states["ifo002"]["position_w"], 
						           "ifo3":states["ifo003"]["position_w"], }
            estiamated_SE3={"ifo2":(#轨迹起始时刻的位姿矩阵放在这里，即ifo2在冻结时刻坐标系位姿R,t)
				            "ifo3":(#轨迹起始时刻的位姿矩阵放在这里，即ifo3在冻结时刻坐标系位姿R,t)}

````



