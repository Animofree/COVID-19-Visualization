<template>
  <div>
    <div style="text-align: left">
        <div style="height: 40px; line-height: 40px">
            <label style="font-weight: 800; font-size: 20px">{{title}}</label>
<!--            <label style="float: right; color: #33FFFF">数据更新时间： {{updateTime}}</label>-->
        </div>
    </div>
    <!-- 总体数据汇总情况 -->
    <el-card class="box-card" style="background: #f4f4f5; ">
        <el-row :gutter="20">
            <el-col :span="6" v-for="item in sums" :key="item.name">
                <div class="grid-content ">
                    <div class="sum_numb" :style="{color: item.color, fontSize: '30px', marginBottom: '8px'}">{{item.sum}}</div>
                    <div class="sum_numb" style="color: #333333; font-size: 14px">{{item.text}}病例</div>
                    <div class="sum_numb" style="font-size: 13px; color: #999999; margin-top: 8px">
                        <label style="font-weight:200">较昨日 </label>
                        <label :style="{color: item.color}">{{item.add}}</label>
                    </div>
                </div>
            </el-col>
        </el-row>
    </el-card>
  </div>
</template>

<script>

export default {
    name: 'Summary',
    props: ['title', 'updateTime', 'sums']
}
</script>

<!-- 添加"scoped"属性以限制CSS -->
<style scoped>

.grid-content {
    border-radius: 4px;
    min-height: 46px;
  }
  .sum_numb {
      height: 20px;
      font-weight: 700;
  }


</style>
