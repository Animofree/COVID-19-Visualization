<template>
    <div class="Hello">
        <Common :title="loader.title"
                :updateTime="loader.updateTime"
                :sums="loader.sums"
                :tabs="tabs"
                :activeName_="loader.activeName"
                @handleClickTab="loader.handleClickTab($event)">
        </Common>

            <el-image :src="require('../assets/1.png')" fit="cover" style="margin-top :0px"></el-image>
            <el-image :src="require('../assets/2.png')" fit="cover"></el-image>
            <el-image :src="require('../assets/3.png')" fit="cover"></el-image>
            <el-image :src="require('../assets/4.png')" fit="cover"></el-image>

    </div>

</template>

<script>
    import Common from './Common.vue';
    import Loader from '../js/common.js'
    import {Utils} from "../js/utils";
    // import img1 from '../assets/1.png'
    // import img2 from '../assets/2.png'
    // import img3 from '../assets/3.png'
    // import img4 from '../assets/4.png'

    export default {
        name: 'Home',
        components: {Common},
        props: {
            msg: String
        },
        data() {
            return {
                title: "国内疫情",
                updateTime: '2020.04.15 02:29',
                sums: [
                    // 2月份的数据
                    {name: 'confirmed', text: '确诊', color: Utils.Colors[0], sum: 63951, add: "+19"},
                    {name: 'suspected', text: '疑似', color: Utils.Colors[1], sum: 8228, add: ""},
                    {name: 'die', text: '死亡', color: Utils.Colors[2], sum: 1382, add: "+1"},
                    {name: 'ok', text: '治愈', color: Utils.Colors[3], sum: 7094, add: "+366"}
                ],

                tabs: [
                    {
                        label: "全国实时疫情", name: 'china', ids: ['ecChina', 'ecBar1'], level: 1,
                        allTime: 0, data: null, mapName: 'china'
                    },
                    {
                        label: "全国疫情回放", name: 'chinaTime', ids: ['ecChinaTime', 'ecBarTime1'], level: 1,
                        allTime: 1, data: null, mapName: "china"
                    },
                    {
                        label: "各省市疫情", name: 'province', ids: ['ecProvince', 'ecBar2'], level: 2,
                        allTime: 0, data: null, mapName: "420000"
                    },
                    {
                        label: "曲线分析", name: "lineChina", ids: ['ecLineChina'], level: 1, isLine: 1,
                        allTime: 1, data: null, mapName: "china"
                    },
                    {
                        label: "模型预测", name: "prediction",
                        ids: ['imgUrl'],

                        level: 1,
                        allTime: 1,
                        data: null,

                    }
                ],


                loader: Loader,
            }
        },
        mounted() {
            Loader.init(this.title, this.updateTime, this.sums, this.tabs);
            [Loader.level, Loader.code] = [1, "86"];
            Loader.loadSummary();
            this.init();
        },
        // 修改初始化页面
        methods: {
            init() {
                Loader.activeName = "lineChina";
                Loader.loadData(this.tabs[1]);
            }
        }
    }

</script>

<!--添加“scoped”属性限制CSS -->
<style scoped>
    h3 {
        margin: 0px 0 16px 0;
    }

    a {
        color: #42b983;
    }

    .grid-content {
        border-radius: 4px;
        min-height: 46px;
    }

    .sum_numb {
        height: 20px;
        font-weight: 700;

    }

    .chart {
        min-height: 320px;
        margin-bottom: 0px;
    }

    /* 定义keyframe动画，命名为blink */
    @keyframes blink {
        0% {
            opacity: 1;
        }
        100% {
            opacity: 0;
        }
    }

    /* 定义blink类*/
    .blink {
        color: #dd4814;
        animation: blink 1s linear infinite;
    }

    .blink > a {
        color: #dd4814 !important
    }

</style>
