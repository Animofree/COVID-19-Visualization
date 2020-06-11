const path = require('path');

module.exports = {
  pages: {
    index: {
      entry: "src/main.js",
      template: "public/index.html",
      filename: "index.html"
    }
  },
  publicPath: '',
  outputDir: 'dist', //打包输出目录默认dist
  configureWebpack : {
    performance: {
      hints:'warning',

      //只给出 js 文件的性能提示
      assetFilter: function(assetFilename) {
        return assetFilename.endsWith('.js');
      }
    }
  },

  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:9400/',
        ws: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  }

};