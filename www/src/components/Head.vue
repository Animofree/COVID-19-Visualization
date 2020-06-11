<template>
  <div>
    <div style="text-align: center; " class="pageHead">
      <div style="font-size: 30px">COVID-19疫情可视化系统</div>
    </div>
  </div>
</template>


<script>

export default {
  name: 'Head',
}
</script>

<style>
html, body {
  height: 100%;
  margin: 0px;
}
a, a>label { cursor: pointer }

/*控制“国内疫情”模块*/
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  height: 100%;
}
/*修改title.png的大小*/
.pageHead {
  height: 100px;
  padding: 10px;
  background: url(../assets/title.png) center top / 100% no-repeat;
  color: #fff
}

</style>
