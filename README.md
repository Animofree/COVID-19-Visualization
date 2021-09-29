# COVID-19-Visualization
基于Python的疫情传播模拟系统

2019年底-2020年初国内新型冠状病毒疫情数据可视化、疫情历史数据分析、数据更新、数据清洗、行政区域代码标准化、趋势预测等

针对此次新型冠状病毒，全国（含武汉）范围内疫情数据变化情况，做可视分析与趋势预测。

前后分离，具体结构见[项目结构]



## 项目特点

1. 支持常规省、市两级地图数据可视化，下钻交互。

2. 动态效果播放呈现各级区域疫情数据随时间变化趋势。

3. 全国省市数据heatmap及时间序列变化趋势。

4. 交互式数据分析。



## 项目结构

1.前端源码：web目录下（VUE、ElementUI、ECharts）

	开发部署：
		
		cd web/epidemic-map
		
		npm install
		
		npm run serve
		
		效果：http://localhost:8080/
	
	

2.后端源码：src目录下（PYTHON3、Flask、MySQL）

	配置文件：src/config.py	
	
	启动服务：
		start.bat - Windows
		

3.数据库：（MySQL）
	
	文件：src/db/epidemic.sql
	
	新建数据库epidemic，将该文件导入MySQL即可。
	
	数据库账号密码在src/config.py中配置
	

4.数据：
	
	使用API：https://lab.isaaclin.cn/nCoV
	
	数据更新：src/data/dxy_record.py （手动）
	
	数据清洗：地区标准化 - region_recognition.py
	
	自动更新任务：startData.bat
	

