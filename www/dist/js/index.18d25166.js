(function (e) {
    function t(t) {
        for (var r, o, s = t[0], l = t[1], u = t[2], d = 0, h = []; d < s.length; d++) o = s[d], Object.prototype.hasOwnProperty.call(i, o) && i[o] && h.push(i[o][0]), i[o] = 0;
        for (r in l) Object.prototype.hasOwnProperty.call(l, r) && (e[r] = l[r]);
        c && c(t);
        while (h.length) h.shift()();
        return n.push.apply(n, u || []), a()
    }

    function a() {
        for (var e, t = 0; t < n.length; t++) {
            for (var a = n[t], r = !0, s = 1; s < a.length; s++) {
                var l = a[s];
                0 !== i[l] && (r = !1)
            }
            r && (n.splice(t--, 1), e = o(o.s = a[0]))
        }
        return e
    }

    var r = {}, i = {index: 0}, n = [];

    function o(t) {
        if (r[t]) return r[t].exports;
        var a = r[t] = {i: t, l: !1, exports: {}};
        return e[t].call(a.exports, a, a.exports, o), a.l = !0, a.exports
    }

    o.m = e, o.c = r, o.d = function (e, t, a) {
        o.o(e, t) || Object.defineProperty(e, t, {enumerable: !0, get: a})
    }, o.r = function (e) {
        "undefined" !== typeof Symbol && Symbol.toStringTag && Object.defineProperty(e, Symbol.toStringTag, {value: "Module"}), Object.defineProperty(e, "__esModule", {value: !0})
    }, o.t = function (e, t) {
        if (1 & t && (e = o(e)), 8 & t) return e;
        if (4 & t && "object" === typeof e && e && e.__esModule) return e;
        var a = Object.create(null);
        if (o.r(a), Object.defineProperty(a, "default", {
            enumerable: !0,
            value: e
        }), 2 & t && "string" != typeof e) for (var r in e) o.d(a, r, function (t) {
            return e[t]
        }.bind(null, r));
        return a
    }, o.n = function (e) {
        var t = e && e.__esModule ? function () {
            return e["default"]
        } : function () {
            return e
        };
        return o.d(t, "a", t), t
    }, o.o = function (e, t) {
        return Object.prototype.hasOwnProperty.call(e, t)
    }, o.p = "";
    var s = window["webpackJsonp"] = window["webpackJsonp"] || [], l = s.push.bind(s);
    s.push = t, s = s.slice();
    for (var u = 0; u < s.length; u++) t(s[u]);
    var c = l;
    n.push([0, "chunk-vendors"]), a()
})({
    0: function (e, t, a) {
        e.exports = a("56d7")
    }, "02da": function (e, t, a) {
        "use strict";
        var r = a("b9de"), i = a.n(r);
        i.a
    }, "137b": function (e, t, a) {
        "use strict";
        var r = a("a263"), i = a.n(r);
        i.a
    }, "359c": function (e, t) {
        e.exports = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAFjElEQVR42uSbb2hVdRjHf7tszdXUWduqWW2u1da1hFrWerGihjkTsSEpYa5pMMFqCiX1IgwMIogIBKP5whLD8M+LXpkFqStQGegLuc2hMxbLojVz4uZWrq3vw32OnK7nnnOe3znn7nfogc+L3f3+PM/3nPOc37+TNzU1pSK2KvAkmA9qwX3gNlAMbuEyo2AEXARnmRT4AfRH6VxeBAIkwFPgRfAsqAzY3s/gW/Al6AKTpgpQDl4HbeCuiC7YL+BzsA38YYoAFOxm0A5mqNzYONgBPmRR9I0E0KQAbAajU9Nno+xDgW4cundAPdgNHlBm2BmwBpzUSViiRwZsBMcMCl6xL+RTB/sYSQ4oArvAC8ps2w9eBmNhCjAbfA2eUPGw42AJuByGABT8YfCIipdRPmjyEsErB9wMDsUweCtRH+IYtASgZPIFaFDxtQaOIU9HgDdAi4q/tXAsohxAye57kO/R+NP8jC0AzeD5HIwGaRT4Fd/epzlHHfGoM8ETsuN+RoKF4JyPUdgph7ol4G3wZwSjvkvcdolDv6d81D/Hsf2nrtMVfgvU+LgShx1+GwYfgE/BVvCq7TEbAj/y9PY3/vsa/68AlII7efo8n/9WPPvbDrZw+05Gd8DDHv7WcGxb3e6AajDu84q0+Rhr14OXQKXGOL2S69b7KNvm0+dxjvF63cyGOgW3ZHOAiVTYNAv87rTXtb8F7gFrBcmowKBMf5Og7FqO9YbX4CZhUGUGCVAqvHCbMgWgH1cLO33UIAGkvqy2LrYlwGJe0pLYYwYJIPWlnGO+LsAqYQP0+tpgkAAbbK9Uv7bKLsAiYeX3QLdBAnTf8H73tkXWUJhWU3oEFX8F9/KQ1CSjIXgfmCuok6Q7oFHY0XYDg7fmCJ8I6zQmNNb29ho889snXUskAWoFFWgN/rzBAtAjcEFQvjbhc+JjWW8M5v9nBGVrSIA5ggq/x0CAQUHZW0mAmYIKEzEQQDIeKCYBCgUVZsVAgNmCsoXSnaGZMRBA5CMJMCYoPy8GAlQJyo6RAFcEFeiwQ5HBwRcp2YGMKyTAgKBCvmHT4ExbqLxXsu02QAL0CztZarAAzwnL9yeEEyGylUq+rZ4LS7BvEutJaExr52l0lAtbqZGku2k6XK4xwvsJPASuGhI8bYCmNAS4PcFDR+nRkmqVPqllim3TCJ5iHrSe5YManb4C3jEg+C3si9TSMfMGQV2APbsdYMY0bIYUcd+6Vpe5M3QiQGNnwTKQl4PAqY8W0BfA3xNOW2OtWQp/BxrBclbcbe/wNNgIqiIIvAa8CXpC2GludTonmM/Z/W6HVaDFtvHC/Sq9P++1lEYDrC7OznT4+Rvwl2BC08yrVUmV3tufG1LOGOAkPmHPARbrXU5krrGVK/N5hsCyXcLHg8rujOh06Xq33eF8kHKpvM5WdiH4x0eHF8EsjVue6gyFHHyKY8wqANHksb+etJV910enHwd47j8KWYCmzD6ydex2TuBgxq36vkenSwMIsCTE4Dud+sjWcbHHa2ZBRvnHwW7OC8PgPDjA2bYwgADVIQXfxzH5FoB4EIxkaXBvjgY7JSEEP8KxOPbhNq2l11ercv5EhWZecThDOMkxpLKW8HEV2rMoexWsMPwOaPfqw68jHS6d7AcNEQ2DgwjQEfYXI/RFxk6XNbchvtUu8+oMrTPU8SrtsOYtXAIuCevQCG+dSn/R4mnST2boc7h9SnacZk4OBRjk/NQlWUeTGDVMR+ePGpjwjrJvXZJKOoubtP38DHhNpb/4nG4jHzrYpwvi2gGTVAVPWiZdktEdAdovc2mX+vyMfdCOIaxsneQZ398OjpYFaLfUob1r3FcyDN/Dfm1V8JH2XnZ2Twht7uG2erntijB9juLj6VhZQv3P7V8BBgAKqy0Tg0JzXwAAAABJRU5ErkJggg=="
    }, 5490: function (e, t, a) {
        "use strict";
        var r = a("a6b6"), i = a.n(r);
        i.a
    }, "56d7": function (e, t, a) {
        "use strict";
        a.r(t);
        a("e260"), a("e6cf"), a("cca6"), a("a79d");
        var r = a("2b0e"), i = function () {
                var e = this, t = e.$createElement, a = e._self._c || t;
                return a("div", {attrs: {id: "app"}}, [a("Head"), a("router-view")], 1)
            }, n = [], o = function () {
                var e = this, t = e.$createElement;
                e._self._c;
                return e._m(0)
            }, s = [function () {
                var e = this, t = e.$createElement, r = e._self._c || t;
                return r("div", [r("div", {
                    staticClass: "pageHead",
                    staticStyle: {"text-align": "left"}
                }, [r("div", {staticStyle: {"text-align": "right"}}, [r("span", [r("a", {
                    staticStyle: {
                        color: "#fff",
                        "text-decoration": "none"
                    }, attrs: {href: "https://github.com/simonblowsnow/2019-ncov-vis"}
                }, [r("img", {
                    staticStyle: {"margin-bottom": "-2px"},
                    attrs: {src: a("359c"), width: "18"}
                }), r("label", {staticStyle: {"font-size": "16px"}}, [e._v(" Github")])])])]), r("div", [r("div", {staticStyle: {"font-size": "24px"}}, [e._v("新冠肺炎疫情"), r("label", {staticStyle: {"font-size": "22px"}}, [e._v(" · 数据分析系统")])]), r("div", {
                    staticStyle: {
                        "font-size": "20px",
                        "margin-top": "5px",
                        color: "#ddd",
                        "font-family": "'Times New Roman', Times, serif"
                    }
                }, [e._v(" WuHan COVID-19 Data Visualization Analysis System ")])])])])
            }], l = {name: "Head"}, u = l, c = (a("e4bd"), a("2877")), d = Object(c["a"])(u, o, s, !1, null, null, null),
            h = d.exports, m = {name: "App", components: {Head: h}}, p = m,
            f = Object(c["a"])(p, i, n, !1, null, null, null), v = f.exports, g = a("313e"), x = a.n(g),
            b = (a("0fae"), a("5453"), a("5c96")), y = a.n(b), S = a("8c4f"), w = function () {
                var e = this, t = e.$createElement, a = e._self._c || t;
                return a("div", {staticClass: "hello"}, [a("el-main", [a("Common", {
                    attrs: {
                        title: e.loader.title,
                        updateTime: e.loader.updateTime,
                        sums: e.loader.sums,
                        tabs: e.tabs,
                        activeName_: e.loader.activeName
                    }, on: {
                        handleClickTab: function (t) {
                            return e.loader.handleClickTab(t)
                        }
                    }
                })], 1), a("div", {
                    staticClass: "blink",
                    staticStyle: {position: "fixed", right: "10px", top: "90px", "font-weight": "400", "font-family": "宋体"}
                }, [a("router-link", {attrs: {to: "/map"}}, [e._v("疫情小区")])], 1)], 1)
            }, T = [], C = function () {
                var e = this, t = e.$createElement, a = e._self._c || t;
                return a("div", [a("div", {staticClass: "hello"}, [a("div", {staticStyle: {"text-align": "left"}}, [a("div", {
                    staticStyle: {
                        height: "40px",
                        "line-height": "40px"
                    }
                }, [a("label", {
                    staticStyle: {
                        "font-weight": "800",
                        "font-size": "18px"
                    }
                }, [e._v(e._s(e.title))]), a("label", {
                    staticStyle: {
                        float: "right",
                        color: "#4197FD"
                    }
                }, [e._v("数据更新时间： " + e._s(e.updateTime))])])]), a("el-card", {
                    staticClass: "box-card",
                    staticStyle: {background: "#f4f4f5"}
                }, [a("el-row", {attrs: {gutter: 20}}, e._l(e.sums, (function (t) {
                    return a("el-col", {
                        key: t.name,
                        attrs: {span: 6}
                    }, [a("div", {staticClass: "grid-content "}, [a("div", {
                        staticClass: "sum_numb",
                        style: {color: t.color, fontSize: "20px", marginBottom: "8px"}
                    }, [e._v(e._s(t.sum))]), a("div", {
                        staticClass: "sum_numb",
                        staticStyle: {color: "#333333", "font-size": "14px"}
                    }, [e._v(e._s(t.text) + "病例")]), a("div", {
                        staticClass: "sum_numb",
                        staticStyle: {"font-size": "13px", color: "#999999", "margin-top": "8px"}
                    }, [a("label", {staticStyle: {"font-weight": "200"}}, [e._v("昨日 ")]), a("label", {style: {color: t.color}}, [e._v(e._s(t.add))])])])])
                })), 1)], 1)], 1), a("div", {staticStyle: {"padding-top": "10px"}}, [a("el-tabs", {
                    on: {"tab-click": e.handleClickTab},
                    model: {
                        value: e.activeName, callback: function (t) {
                            e.activeName = t
                        }, expression: "activeName"
                    }
                }, e._l(e.tabs, (function (t, r) {
                    return a("el-tab-pane", {
                        key: r,
                        attrs: {label: t.label, name: t.name}
                    }, [r < 4 ? a("el-row", {
                        directives: [{
                            name: "show",
                            rawName: "v-show",
                            value: e.activeName == t.name,
                            expression: "activeName==c.name"
                        }], attrs: {gutter: 5}
                    }, [a("el-col", {attrs: {xs: 24, sm: 24, md: 12, lg: 12, xl: 12}}, [a("div", {
                        staticClass: "chart",
                        style: {height: e.mapHeight},
                        attrs: {id: t.ids[0]}
                    })]), a("el-col", {attrs: {xs: 24, sm: 24, md: 12, lg: 12, xl: 12}}, [a("div", {
                        staticClass: "chart",
                        staticStyle: {height: "500px"}
                    }, [a("el-scrollbar", {staticStyle: {height: "100%"}}, [a("div", {
                        staticStyle: {height: "900px"},
                        attrs: {id: t.ids[1]}
                    })])], 1)])], 1) : a("el-row", {
                        directives: [{
                            name: "show",
                            rawName: "v-show",
                            value: e.activeName == t.name,
                            expression: "activeName==c.name"
                        }]
                    }, [a("el-col", {attrs: {xs: 24, sm: 24, md: 24, lg: 24, xl: 24}}, [a("div", {
                        staticClass: "chart",
                        staticStyle: {height: "500px"},
                        attrs: {id: t.ids[0]}
                    })])], 1)], 1)
                })), 1)], 1), a("div", {staticStyle: {height: "20px"}})])
            }, M = [],
            A = (a("d81d"), a("13d5"), a("fb6a"), a("b0c0"), a("d3b7"), a("4d63"), a("ac1f"), a("25f0"), a("466d"), a("5319"), a("bc3a")),
            P = a.n(A), D = {
                Test: "test",
                Login: "webLogin",
                GetMap: "getMap",
                GetDataDetails: "getDataDetails",
                GetTimeData: "getTimeData",
                GetDataChina: "getDataChina",
                GetDataSummary: "getDataSummary",
                GetDataPos: "getDataPos"
            }, B = a("125e"), F = {
                setCookie: function (e, t, a) {
                    a = a || 200;
                    var r = new Date;
                    r.setTime(r.getTime() + 3600 * a * 1e3), document.cookie = e + "=" + escape(t) + ";expires=" + r.toUTCString()
                }, getCookie: function (e, t) {
                    void 0 === t && (t = null);
                    var a = new RegExp("(^| )" + e + "=([^;]*)(;|$)"), r = document.cookie.match(a);
                    return r ? unescape(r[2]) : t
                }, postData: function (e, t, a, r, i) {
                    t["tk"] = F.getCookie("tk"), t["lc"] = F.getCookie("lc"), P.a.post("/api/" + e, t, i || {}).then((function (e) {
                        e.data.error ? alert(e.data.message) : a && a(e.data)
                    })).catch((function (e) {
                        r && r(e)
                    }))
                }, ajaxData: function (e, t, a, r) {
                    t["tk"] = F.getCookie("tk"), t["lc"] = F.getCookie("lc"), P.a.get("/api/" + e, {params: t}).then((function (e) {
                        e.data.error ? alert(e.data.message) : a && a(e.data)
                    })).catch((function (e) {
                        r && r(e)
                    }))
                }, login: function (e, t, a) {
                    F.postData(D.Login, {username: e || "david", password: t || "123456"}, (function (e) {
                        e.error || (F.setCookie("tk", e.data.tk), F.setCookie("lc", e.data.lc), a && a())
                    }))
                }, Names: {}, formatRegion: function (e, t) {
                    var a = x.a.getMap(e);
                    if (a) {
                        var r = F.Names[e];
                        return t.map((function (e) {
                            return {name: r[e[0]], value: e[1], code: e[0], tags: e.slice(2)}
                        }))
                    }
                }, replaceAll: function (e, t, a) {
                    return e.replace(new RegExp(t, "gm"), a)
                }, registerMap: function (e, t) {
                    t = JSON.parse(t), x.a.registerMap(e, t), F.Names[e] = t.features.reduce((function (e, t) {
                        return e[t.id.toString()] = t.properties.name, e
                    }), {})
                }, drawGraph: function (e, t) {
                    var a = x.a.init(document.getElementById(t));
                    return a.setOption(e), a
                }, draw: function (e, t) {
                    return e.instance = F.drawGraph(e.option, t), e.instance
                }, getDevice: function () {
                    var e = document.documentElement.offsetWidth || document.body.offsetWidth;
                    return e < 768 ? "xs" : e < 1064 ? "sm" : e < 1200 ? "md" : "lg"
                }, last: function (e) {
                    return e[e.length - 1]
                }, interpolateColor: function (e, t, a) {
                    if (void 0 == a) {
                        var r = [t, 0];
                        a = r[0], t = r[1]
                    }
                    for (var i = a - t, n = 1 / (e.length - 1), o = [], s = 0; s < e.length - 1; s++) o.push(B["a"](e[s], e[s + 1]));
                    this.compute = function (e) {
                        if (e >= a) return o[o.length - 1](1);
                        var r = (e - t) / i, s = parseInt(r / n), l = (r - s * n) / n, u = o[s];
                        return u(l)
                    }
                }, Colors: ["#F55253", "#FF961E", "#66666c", "#178B50"]
            }, k = {
                name: "Common",
                props: {title: String, updateTime: String, sums: Array, tabs: Array, activeName_: String},
                data: function () {
                    return {activeName: "", mapHeight: "xs" === F.getDevice() ? "330px" : "500px"}
                },
                watch: {
                    activeName_: function (e) {
                        this.activeName = e
                    }
                },
                methods: {
                    handleClickTab: function (e) {
                        this.$emit("handleClickTab", parseInt(e.index))
                    }
                }
            }, O = k, E = (a("02da"), Object(c["a"])(O, C, M, !1, null, "7b3ab4ea", null)), R = E.exports;
        a("b64b");

        function I() {
            return {
                data: ["2016", "2017"],
                axisType: "category",
                autoPlay: !0,
                loop: !0,
                currentIndex: 0,
                playInterval: 1200,
                left: "0",
                right: "1%",
                top: "0%",
                width: "99%",
                symbolSize: 10,
                label: {
                    formatter: function (e) {
                        return e.substr(5)
                    }
                },
                checkpointStyle: {borderColor: "#777", borderWidth: 2},
                controlStyle: {
                    showNextBtn: !0,
                    showPrevBtn: !0,
                    normal: {color: "#ff8800", borderColor: "#ff8800"},
                    emphasis: {color: "#aaa", borderColor: "#aaa"}
                }
            }
        }

        var z = I, j = {
                title: {text: "", top: 15, subtext: "点击各区块查看下级地图", textStyle: {color: "#4197FD", fontSize: 16}},
                visualMap: {
                    min: 0,
                    left: "left",
                    top: "bottom",
                    text: ["高", "低"],
                    calculable: !0,
                    inRange: {color: ["#FFAA85", "#FF7B69", "#BF2121", "#7F1818"], symbolSize: [40, 40]}
                },
                series: [{
                    name: "china",
                    type: "map",
                    mapType: "china",
                    roam: !1,
                    label: {
                        emphasis: {
                            show: !0,
                            formatter: function (e) {
                                var t = e.data ? e.data.tags[2] : 0;
                                return "{fline|" + e.name + "}\n{tline|确诊: " + e.value + "\n死亡：" + t + "}\n"
                            },
                            position: "top",
                            align: "left",
                            fontSize: 14,
                            width: 150,
                            backgroundColor: "rgba(50, 50, 50, 0.8)",
                            padding: [0, 0],
                            borderRadius: 3,
                            color: "#f7fafb",
                            rich: {
                                fline: {
                                    padding: [0, 5, 5, 10],
                                    height: 20,
                                    fontSize: 16,
                                    fontWeight: 400,
                                    color: "#FFFFFF"
                                }, tline: {padding: [0, 5, 5, 10], color: "#F55253"}
                            }
                        }, normal: {show: !1}
                    },
                    itemStyle: {
                        normal: {label: {show: !1}},
                        emphasis: {
                            areaColor: null,
                            borderColor: "#BF2121",
                            borderWidth: 1.5,
                            shadowColor: "red",
                            shadowOffsetX: -1,
                            shadowOffsetY: -1,
                            label: {show: !0}
                        }
                    },
                    data: [{name: "北京", value: Math.round(1e3 * Math.random())}]
                }]
            }, _ = [], U = {baseOption: {timeline: z(), visualMap: j.visualMap, series: j.series}, options: []},
            H = {name: "china", option: j, superOption: U, initData: null, instance: null, useMaxValue: !0};

        function N(e, t, a) {
            var r = F.formatRegion(t, e), i = r.reduce((function (e, t) {
                return e + t.value
            }), 0);
            return r.sort((function (e, t) {
                return t.value - e.value
            })), a.title.text = "共确诊：" + i, r.length > 0 && (a.title.subtext = r[0].name + "：" + r[0].value), _.push(r.length < 2 ? 10 : r[1].value), a.series[0]["data"] = r, H.useMaxValue || (a["visualMap"] = {max: parseInt(1.25 * _[_.length - 1])}), a
        }

        function L(e, t) {
            var a = Object.keys(e);
            return U.baseOption.timeline.data = a, U.options = a.map((function (a) {
                var r = {title: {text: "", top: 55, textStyle: {color: "#bbb", fontSize: 16}}, series: [{}]};
                return N(e[a], t, r)
            })), U.baseOption.timeline.autoPlay = !1, U.baseOption.timeline.currentIndex = a.length - 1, U
        }

        H.initData = function (e, t, a, r) {
            a = a || "china", j.series[0].mapType = a, j.series[0].label.normal.show = "china" != a, _ = [];
            var i = j;
            i = r ? L(e, a) : N(e, a, i);
            var n = parseInt(1.25 * _.reduce((function (e, t) {
                return e > t ? e : t
            })));
            return j.visualMap.max = n, H.instance = F.drawGraph(i, t), H.instance
        };
        var G = H, W = G, V = (a("4160"), a("159b"), {
            tooltip: {trigger: "axis", axisPointer: {type: "shadow"}},
            title: {text: "", top: 15, textStyle: {color: "#4197FD", fontSize: 16}},
            legend: {show: !0, right: 10, top: 50, data: ["确诊", "疑似", "死亡", "治愈"]},
            grid: [{left: -65, top: "88", bottom: "3%", width: 100, containLabel: !0}, {
                left: 45,
                width: "58%",
                top: "68",
                bottom: "3%",
                containLabel: !0
            }, {right: "40", width: "25%", top: "68", bottom: "3%", containLabel: !0}],
            xAxis: [{left: "10px", show: !1}, {type: "value", show: !1, position: "top", gridIndex: 1}, {
                type: "value",
                show: !1,
                position: "top",
                gridIndex: 2
            }],
            yAxis: [{
                type: "category",
                inverse: !0,
                data: ["福建", "广州", "厦门", "南宁", "北京", "长沙", "重庆"],
                axisLine: {show: !1},
                axisTick: {show: !1},
                axisLabel: {
                    interval: 0,
                    margin: 85,
                    textStyle: {color: "#455A74", align: "left", fontSize: 14},
                    rich: {
                        a: {
                            color: "#fff",
                            backgroundColor: "#FAAA39",
                            width: 20,
                            height: 20,
                            align: "center",
                            borderRadius: 2
                        },
                        b: {
                            color: "#fff",
                            backgroundColor: "#4197FD",
                            width: 20,
                            height: 20,
                            align: "center",
                            borderRadius: 2
                        }
                    },
                    formatter: function (e, t) {
                        return "{" + (t < 3 ? "a" : "b") + "|" + (t + 1) + "}  " + e.substr(0, 4)
                    }
                }
            }, {
                gridIndex: 1,
                type: "category",
                inverse: !0,
                show: !1,
                data: ["福建", "广州", "厦门", "南宁", "北京", "长沙", "重庆"],
                axisLine: {show: !1},
                axisTick: {show: !1}
            }, {
                gridIndex: 2,
                type: "category",
                inverse: !0,
                show: !1,
                data: ["福建", "广州", "厦门", "南宁", "北京", "长沙", "重庆"],
                axisLine: {show: !1},
                axisTick: {show: !1}
            }],
            series: [{
                name: "确诊",
                type: "bar",
                barWidth: 20,
                xAxisIndex: 1,
                yAxisIndex: 1,
                data: [320, 302, 301, 334, 390, 330, 320]
            }, {
                name: "疑似",
                type: "bar",
                stack: !0,
                barWidth: 20,
                xAxisIndex: 2,
                yAxisIndex: 2,
                data: [200, 302, 301, 334, 390, 330, 320]
            }]
        }), Q = ["确诊", "疑似", "死亡", "治愈"], Y = [0, 0, 0, 0], X = {
            baseOption: {
                timeline: z(),
                legend: V.legend,
                tooltip: V.tooltip,
                grid: V.grid,
                xAxis: V.xAxis,
                yAxis: V.yAxis,
                series: V.series,
                animationDurationUpdate: 1200,
                animationEasingUpdate: "quinticInOut"
            }, options: []
        }, q = {name: "barChina", option: V, superOption: X, initData: null, instance: null, useMaxValue: !0};

        function K(e, t, a) {
            var r = [[], [], [], []], i = 0;
            e.sort((function (e, t) {
                return t[1] - e[1]
            })).forEach((function (e) {
                [1, 3, 4, 5].forEach((function (t, a) {
                    r[a].push(e[t]), e[t] > Y[a] && (Y[a] = e[t]), 0 == a && (i += e[t])
                }))
            }));
            for (var n = 0; n < 3; n++) a["yAxis"][n]["data"] = e.map((function (e) {
                return t[e[0]] || e[2]
            }));
            return a.title.text = "累计确诊 " + i, a["series"] = Q.map((function (e, t) {
                return {
                    name: Q[t],
                    type: "bar",
                    barWidth: 20,
                    stack: !(t < 1),
                    xAxisIndex: t < 1 ? 1 : 2,
                    yAxisIndex: t < 1 ? 1 : 2,
                    itemStyle: {
                        normal: {
                            barBorderRadius: [2, 2, 2, 2],
                            shadowBlur: [0, 0, 0, 10],
                            color: F.Colors[t],
                            shadowColor: F.Colors[t]
                        }
                    },
                    label: {normal: {show: t < 1 || 3 == t, distance: 5, position: "right"}},
                    data: r[t]
                }
            })), a
        }

        function J(e, t) {
            var a = Object.keys(e);
            return X.baseOption.timeline.data = a, X.options = a.map((function (a) {
                var r = {
                    title: {text: "", top: 65, textStyle: {color: "#bbb", fontSize: 14}},
                    yAxis: [{data: []}, {data: []}, {data: []}]
                };
                return K(e[a], t, r)
            })), X
        }

        function Z(e, t, a) {
            var r = t;
            if (a) {
                var i = Object.keys(t);
                r = t[i[i.length - 1]]
            }
            var n = 26 * r.length + 20;
            document.getElementById(e).style.height = (n < 350 ? 350 : n) + "px"
        }

        q.initData = function (e, t, a, r) {
            var i = V;
            i["legend"]["data"] = Q, Y = [0, 0, 0, 0], i = r ? J(e, a) : K(e, a, i), "xs" === F.getDevice() && (V.grid[1].width = "51%"), q.useMaxValue && (V.xAxis[1]["max"] = Y[0], V.xAxis[2]["max"] = Y[2] + Y[3]), Z(t, e, r);
            var n = F.drawGraph(i, t);
            n.dispatchAction({type: "legendUnSelect", name: "疑似"})
        };
        var $ = q, ee = $, te = a("3835");

        function ae() {
            var e = {
                title: [{text: "", left: 0, textStyle: {fontSize: 16}}, {
                    text: "",
                    left: 100,
                    textStyle: {color: "#F55253", fontWeight: "normal", fontSize: 12}
                }],
                tooltip: {
                    trigger: "axis",
                    backgroundColor: "rgba(13,177,205,0.8)",
                    axisPointer: {
                        type: "shadow",
                        label: {show: !0, backgroundColor: "#7B7DDC"},
                        shadowStyle: {color: "rgba(250, 250, 250, 0.25)"}
                    }
                },
                axisPointer: {link: {}},
                legend: {data: ["累计确诊", "新增确诊", "累计治愈"], top: 0, right: "30"},
                grid: [{containLabel: !0, left: 0, right: 0, top: 30, height: 120}],
                xAxis: [{
                    data: ["02-01", "02-02", "02-03"],
                    axisLine: {lineStyle: {color: "#999", width: .6}},
                    axisTick: {show: !0},
                    axisLabel: {
                        formatter: function (e) {
                            return e.substr(5)
                        }
                    }
                }],
                yAxis: [{
                    name: "", axisLine: {lineStyle: {color: "#666"}}, axisLabel: {
                        formatter: function (e) {
                            return e < 5e3 ? e : e / 1e3 + "k"
                        }, textStyle: {color: "#666"}
                    }, splitLine: {show: !0, lineStyle: {color: "rgba(205,205,205,0.3)"}}
                }, {
                    name: "",
                    splitLine: {show: !1},
                    min: 0,
                    axisLine: {lineStyle: {color: "#B4B4B4"}},
                    axisLabel: {
                        formatter: function (e) {
                            return e < 1e3 ? e : e / 1e3 + "k"
                        }
                    }
                }],
                series: [{
                    name: "累计确诊",
                    type: "line",
                    smooth: !0,
                    showAllSymbol: !0,
                    symbol: "emptyCircle",
                    symbolSize: 4,
                    yAxisIndex: 0,
                    itemStyle: {normal: {color: "#F55253"}},
                    lineStyle: {width: 1},
                    data: [10, 20, 50]
                }, {
                    name: "新增确诊",
                    type: "bar",
                    barMaxWidth: 20,
                    yAxisIndex: 1,
                    itemStyle: {normal: {barBorderRadius: 3, color: "#F55253"}},
                    data: [10, 8, 5]
                }, {
                    name: "累计治愈",
                    type: "line",
                    smooth: !0,
                    showAllSymbol: !0,
                    symbol: "roundRect",
                    symbolSize: 4,
                    yAxisIndex: 1,
                    itemStyle: {normal: {color: "#178B50"}},
                    lineStyle: {width: 1},
                    data: [10, 20, 50]
                }]
            };
            return e
        }

        var re = ae(), ie = [10, 10],
            ne = {name: "lineChina", option: re, initData: null, instance: null, useMaxValue: !0};

        function oe(e) {
            var t = {}, a = Object.keys(e).sort();
            return a.forEach((function (a) {
                e[a].forEach((function (e) {
                    var r = e[0];
                    r in t || (t[r] = {name: e[2], data: []});
                    var i = [a, e[1], "", e[3], e[4], e[5], e[6], e[7]];
                    t[r].data.push(i)
                }))
            })), console.log(t), t
        }

        function se(e, t) {
            var a = 150 * t + 40;
            document.getElementById(e).style.height = (a < 350 ? 350 : a) + "px"
        }

        function le(e) {
            return e[e.length - 1]
        }

        function ue(e) {
            var t = ["↑", "↓"];
            if (e.length < 2) return "";
            var a = le(e) > e[e.length - 2] ? 0 : 1, r = t[a];
            if (2 === e.length) return r;
            for (var i = e.length - 3; i >= 0; i--) {
                var n = e[i + 1] > e[i] ? 0 : 1;
                if (n != a) break;
                r += t[n]
            }
            return r.length > 5 ? r.substr(0, 5) + "➪" + r.length : r
        }

        function ce(e, t) {
            var a = oe(e), r = Object.keys(a);
            se(t, r.length), r.sort((function (e, t) {
                return le(a[t].data)[1] - le(a[e].data)[1]
            })), r.length > 0 && (ie[0] = le(a[r[0]].data)[1]), r.length > 1 && (ie[1] = le(a[r[1]].data)[1]);
            var i = new F.interpolateColor(["#FFAA85", "#FF7B69", "#BF2121", "#7F1818"], 1.5 * ie[1]),
                n = [[], [], [], [], []];
            return re.title = n[0], re.grid = n[1], re.xAxis = n[2], re.yAxis = n[3], re.series = n[4], r.forEach((function (e, t) {
                var r = ae(), n = [r.title, r.grid[0], r.xAxis[0], r.yAxis, r.series], o = n[0], s = n[1], l = n[2],
                    u = n[3], c = n[4];
                o[0].text = a[e].name, o[0].top = 150 * t + 40, o[1].top = 150 * t + 45, s.top = 150 * t + 70, l.gridIndex = t, u[0].gridIndex = u[1].gridIndex = t, c[0].xAxisIndex = c[1].xAxisIndex = c[2].xAxisIndex = t;
                var d = [2 * t, 2 * t + 1, 2 * t];
                c[0].yAxisIndex = d[0], c[1].yAxisIndex = d[1], c[2].yAxisIndex = d[2];
                var h = a[e].data.reduce((function (e, t, a) {
                    var r = [t[0], t[1], t[6], t[5]];
                    return e[0][a] = r[0], e[1][a] = r[1], e[2][a] = r[2], e[3][a] = r[3], e
                }), [[], [], [], []]), m = Object(te["a"])(h, 4);
                l.data = m[0], c[0].data = m[1], c[1].data = m[2], c[2].data = m[3];
                var p = le(h[1]);
                o[1].text = "确诊 " + p + ", 新增 " + le(h[2]) + " · " + ue(h[2]), o[1].left = 15 * a[e].name.length + 15, c[0].itemStyle.normal.color = c[1].itemStyle.normal.color = i.compute(p), re.title.push(o[0], o[1]), re.grid.push(s), re.xAxis.push(l), re.yAxis.push(u[0], u[1]), re.series.push(c[0], c[1], c[2])
            })), re
        }

        ne.initData = function (e, t, a) {
            var r = ce(e, t);
            console.log([e, t, a, r]);
            var i = F.draw(ne, t);
            return i
        };
        var de = ne, he = de, me = {
                title: "",
                updateTime: "",
                tabs: [],
                sums: [],
                activeName: "china",
                code: 86,
                level: 1,
                init: function (e, t, a, r) {
                    var i = [e, t, a, r];
                    this.title = i[0], this.updateTime = i[1], this.sums = i[2], this.tabs = i[3]
                },
                loadSummary: function () {
                    var e = this;
                    F.ajaxData(D.GetDataSummary, {level: this.level, name: this.code}, (function (t) {
                        e.updateTime = t.data.updateTime;
                        for (var a = t.data.summary, i = 0; i < 4; i++) a[0][i] && r["default"].set(e.sums[i], "sum", a[0][i]), a[1][i] && r["default"].set(e.sums[i], "add", "+" + a[1][i])
                    }))
                },
                loadMap: function (e) {
                    var t = this, a = e.mapName;
                    if (a in F.Names) return t.loadData(e);
                    F.ajaxData(D.GetMap, {id: a}, (function (r) {
                        F.registerMap(a, r.data), t.loadData(e)
                    }))
                },
                loadData: function (e) {
                    var t = this, a = [e.mapName, e.level, e.allTime], r = a[0], i = a[1], n = a[2];
                    if (e.data) return t.drawGraph(e);
                    var o = n ? D.GetTimeData : D.GetDataDetails;
                    F.ajaxData(o, {level: i, name: r}, (function (a) {
                        e.data = a.data, t.drawGraph(e, a.data)
                    }))
                },
                drawGraph: function (e, t) {
                    var a = this, r = [e.mapName, e.level, e.allTime], i = r[0], n = r[1], o = r[2];
                    if (t || (t = e.data), e.isLine) {
                        var s = he.initData(t, e.ids[0], n);
                        s.on("click", (function (e) {
                            console.log(e)
                        }))
                    } else {
                        var l = W.initData(t, e.ids[0], i, o);
                        l.off("click"), l.on("click", (function (e) {
                            if ("map" === e.seriesType) {
                                var t = a.tabs[2 + o], r = [e.data.code, null];
                                t.mapName = r[0], t.data = r[1], a.activeName = t.name, a.loadMap(t)
                            }
                        }));
                        var u = F.Names[i];
                        ee.initData(t, e.ids[1], u, o)
                    }
                },
                handleClickTab: function (e) {
                    var t = this.tabs[e];
                    if (t.isLine) return this.loadData(t);
                    this.loadMap(t)
                }
            }, pe = me, fe = {
                name: "Home", components: {Common: R}, props: {msg: String}, data: function () {
                    return {
                        title: "国内疫情",
                        updateTime: "2020.02.15 02:29",
                        sums: [{
                            name: "confirmed",
                            text: "确诊",
                            color: F.Colors[0],
                            sum: 63951,
                            add: "+19"
                        }, {name: "suspected", text: "疑似", color: F.Colors[1], sum: 8228, add: ""}, {
                            name: "die",
                            text: "死亡",
                            color: F.Colors[2],
                            sum: 1382,
                            add: "+1"
                        }, {name: "ok", text: "治愈", color: F.Colors[3], sum: 7094, add: "+366"}],
                        tabs: [{
                            label: "全国实时疫情",
                            name: "china",
                            ids: ["ecChina", "ecBar1"],
                            level: 1,
                            allTime: 0,
                            data: null,
                            mapName: "china"
                        }, {
                            label: "时间序列回放",
                            name: "chinaTime",
                            ids: ["ecChinaTime", "ecBarTime1"],
                            level: 1,
                            allTime: 1,
                            data: null,
                            mapName: "china"
                        }, {
                            label: "省实时疫情",
                            name: "province",
                            ids: ["ecProvince", "ecBar2"],
                            level: 2,
                            allTime: 0,
                            data: null,
                            mapName: "420000"
                        }, {
                            label: "省舆情回放",
                            name: "provinceTime",
                            ids: ["ecProvinceTime", "ecBarTime2"],
                            level: 2,
                            allTime: 1,
                            data: null,
                            mapName: "420000"
                        }, {
                            label: "曲线分析",
                            name: "lineChina",
                            ids: ["ecLineChina"],
                            level: 1,
                            isLine: 1,
                            allTime: 1,
                            data: null,
                            mapName: "china"
                        }],
                        loader: pe,
                        mapHeight: "xs" === F.getDevice() ? "330px" : "500px"
                    }
                }, mounted: function () {
                    pe.init(this.title, this.updateTime, this.sums, this.tabs);
                    var e = [1, "86"];
                    pe.level = e[0], pe.code = e[1], pe.loadSummary(), this.init()
                }, methods: {
                    init: function () {
                        pe.activeName = "lineChina", pe.loadData(this.tabs[4])
                    }
                }
            }, ve = fe, ge = (a("137b"), Object(c["a"])(ve, w, T, !1, null, "39af770a", null)), xe = ge.exports,
            be = function () {
                var e = this, t = e.$createElement, a = e._self._c || t;
                return a("div", {attrs: {id: "root"}}, [a("div", {
                    staticStyle: {height: "100%"},
                    attrs: {id: "map"}
                }, [e._v("地图")]), a("div", {
                    staticStyle: {
                        position: "fixed",
                        right: "10px",
                        top: "160px",
                        width: "100px"
                    }
                }, [a("el-select", {
                    attrs: {size: "small", placeholder: "请选择"},
                    on: {change: e.loadProvince},
                    model: {
                        value: e.currentProvince, callback: function (t) {
                            e.currentProvince = t
                        }, expression: "currentProvince"
                    }
                }, e._l(e.provinces, (function (e) {
                    return a("el-option", {key: e.code, attrs: {label: e.name, value: e.code}})
                })), 1)], 1), a("div", {
                    staticStyle: {
                        position: "fixed",
                        right: "10px",
                        top: "120px",
                        width: "100px"
                    }
                }, [a("el-select", {
                    attrs: {size: "small", placeholder: "请选择"},
                    on: {change: e.resetMap},
                    model: {
                        value: e.mapIndex, callback: function (t) {
                            e.mapIndex = t
                        }, expression: "mapIndex"
                    }
                }, e._l(e.maps, (function (e) {
                    return a("el-option", {key: e.index, attrs: {label: e.label, value: e.index}})
                })), 1)], 1), a("div", {
                    staticClass: "blink",
                    staticStyle: {
                        position: "fixed",
                        right: "10px",
                        top: "90px",
                        "font-weight": "400",
                        "font-family": "宋体"
                    }
                }, [a("router-link", {attrs: {to: "/china"}}, [e._v("返回")])], 1)])
            }, ye = [],
            Se = (a("a15b"), a("1276"), "pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXFhYTA2bTMyeW44ZG0ybXBkMHkifQ.gUGbDOPUN1v1fTs5SeOR4A"),
            we = {
                Gaode: {
                    name: "高德",
                    urlTemplate: "http://wprd0{s}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8",
                    subdomains: ["1", "2", "3", "4"],
                    attribution: '&copy; <a target="_blank" href="http://ditu.amap.com">高德地图</a>',
                    weixing: "http://wprd0{s}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7"
                },
                Baidu: {
                    name: "百度",
                    spatialReference: {projection: "baidu"},
                    urlTemplate: "http://online{s}.map.bdimg.com/onlinelabel/?qt=tile&x={x}&y={y}&z={z}&styles=pl&scaler=1&p=1",
                    subdomains: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    attribution: '&copy; <a target="_blank" href="http://map.baidu.com">Baidu</a>'
                },
                OpenStreetMap: {
                    name: "OpenStreet",
                    urlTemplate: "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    subdomains: ["a", "b", "c"],
                    attribution: '&copy; <a href="http://osm.org">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/">CARTO</a>',
                    weixing: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
                },
                Mapbox: {
                    name: "Mapbox",
                    urlTemplate: "https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/256/{z}/{x}/{y}?access_token=" + Se,
                    subdomains: ["a", "b", "c", "d"],
                    attribution: '&copy; <a target="_blank" href="http://mapbox.cn">Mapbox</a>',
                    weixing: "https://{s}.tiles.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token=" + Se
                }
            }, Te = [], Ce = ["Gaode", "OpenStreetMap", "Mapbox"];
        Ce.forEach((function (e) {
            Te.push({label: we[e].name, value: e + "_0"}), we[e].weixing && Te.push({
                label: we[e].name + "卫星",
                value: e + "_1"
            })
        }));
        var Me = a("ec60"), Ae = a("5a89"), Pe = a("151c");

        function De(e, t) {
            var a = e.getLayer("building"), r = e.getLayer("label");
            a && (a.remove(), r.remove()), a = new Pe["b"]("building", {
                forceRenderOnMoving: !0,
                forceRenderOnRotating: !0
            }), r = new Me["m"]("label").addTo(e);
            var i = [], n = new Ae["y"]({color: "#F55253", transparent: !0}),
                o = new Ae["y"]({color: "red", transparent: !0});
            a.prepareToDraw = function (e, s) {
                var l = new Ae["i"](16777215);
                l.position.set(0, -10, 10).normalize(), s.add(l), t.features.forEach((function (e) {
                    var t = 50, s = e.properties.levels || 6,
                        l = a.toExtrudePolygon(Me["e"].toGeometry(e), {height: s * t, topColor: "#fff"}, n);
                    l.setInfoWindow(Fe(e)), ["mouseout", "mouseover"].forEach((function (e) {
                        l.on(e, (function (e) {
                            "mouseout" === e.type && this.setSymbol(n), "mouseover" === e.type && this.setSymbol(o)
                        }))
                    })), i.push(l), Be(e).addTo(r)
                })), a.addMesh(i), a.config("animation", !0)
            }, a.addTo(e)
        }

        function Be(e) {
            var t = e.properties.cp, a = e.properties.name, r = [t[0], t[1] - .0022], i = new Me["f"](a, r, {
                draggable: !0,
                textSymbol: {
                    textFaceName: "monospace",
                    textFill: "#34495e",
                    textHaloFill: "#fff",
                    textHaloRadius: 4,
                    textSize: 18,
                    textWeight: "bold",
                    textVerticalAlignment: "top"
                },
                boxStyle: {padding: [12, 8]}
            });
            return i.setInfoWindow(Fe(e)), i.on("click", (function () {
                this.openInfoWindow()
            })), i
        }

        function Fe(e) {
            return {
                single: !0,
                width: 283,
                height: 105,
                custom: !0,
                dx: -3,
                dy: -12,
                content: '<div class="content"><div class="pop_title">' + e.properties.name + '</div><div class="pop_dept">时间：' + e.properties.date + '</div><div class="arrow"></div></div>'
            }
        }

        var ke = {Polygon: 1, MultiPolygon: 1}, Oe = {Polygon: Me["j"], MultiPolygon: Me["i"]}, Ee = {
            $this: null,
            mapType: {map: "Gaode", type: "1"},
            mapConfig: {center: [104.071927, 30.665132], zoom: 14, pitch: 52, baseLayer: null},
            init: function (e) {
                this.$this = e
            },
            getBase: function () {
                var e = F.getCookie("mapStyle");
                if (e) {
                    var t = e.split("_");
                    this.mapType = {map: t[0], type: t[1]}, this.mapTypes = e
                }
                var a = we[this.mapType.map], r = {
                    urlTemplate: parseInt(this.mapType.type) ? a.weixing : a.urlTemplate,
                    subdomains: a.subdomains,
                    attribution: a.attribution,
                    opacity: 1,
                    spatialReference: a["spatialReference"] || null
                };
                return "Gaode" === this.mapType.map && "1" == this.mapType.type && (r["cssFilter"] = "sepia(80%) invert(90%)"), r
            },
            getConfig: function (e) {
                e = e || this.getBase(), this.mapConfig.baseLayer = new Me["k"]("base", e);
                var t = F.getCookie("mapInfo");
                if (t && "undefined" !== t) {
                    var a = t.split("_");
                    this.mapConfig.center = [parseFloat(a[0]), parseFloat(a[1])], this.mapConfig.zoom = parseInt(a[2])
                }
                return "Gaode" === this.mapType.map && "0" == this.mapType.type ? this.mapConfig.pitch = 0 : this.mapConfig.pitch = 52, this.mapConfig
            },
            handleMapMove: function (e) {
                console.log(1);
                var t = e.target, a = t.getCenter(), r = [a.x, a.y, t.getZoom()].join("_");
                F.setCookie("mapInfo", r)
            },
            loadData: function (e, t) {
                F.ajaxData(D.GetDataPos, {code: e}, (function (e) {
                    var a = [];
                    e.data.forEach((function (e) {
                        if (e[1] in ke) {
                            var t = {
                                type: "Feature",
                                properties: {
                                    name: e[0],
                                    cp: "Point" === e[3] ? JSON.parse(e[4]) : [],
                                    levels: 10,
                                    date: e[e.length - 1].substr(0, 10)
                                },
                                geometry: {type: e[1], coordinates: JSON.parse(e[2])}
                            };
                            a.push(t)
                        }
                    })), t({features: a, type: "FeatureCollection"})
                }))
            }
        };

        function Re(e, t) {
            var a = e.getLayer("polygon");
            a || (a = new Me["m"]("polygon").addTo(e));
            var r = {
                polygon: {
                    visible: !0,
                    cursor: "pointer",
                    shadowBlur: 0,
                    shadowColor: "black",
                    draggable: !1,
                    dragShadow: !1,
                    drawOnAxis: null,
                    symbol: {
                        lineColor: "#f00",
                        lineWidth: 1,
                        polygonFill: "#F55253",
                        polygonOpacity: .5,
                        lineColorHighlight: "#f00",
                        polygonFillHighlight: "#f00"
                    }
                }
            }, i = r["polygon"];
            t.features.forEach((function (e) {
                var t = e.geometry.type;
                if (ke[t]) {
                    var r = new Oe[t](e.geometry.coordinates, i);
                    r.addTo(a), Ie(e).addTo(a)
                }
            }))
        }

        function Ie(e) {
            var t = e.properties.cp, a = e.properties.name, r = [t[0], t[1] - .0012], i = new Me["f"](a, r, {
                draggable: !0,
                textSymbol: {
                    textFaceName: "monospace",
                    textFill: "#f00",
                    textHaloFill: "#fff",
                    textHaloRadius: 4,
                    textSize: 18,
                    textWeight: "bold",
                    textVerticalAlignment: "top"
                },
                boxStyle: {padding: [12, 8], symbol: {markerFill: "#FF7F50"}}
            });
            return i.setInfoWindow(Fe(e)), i.on("click", (function () {
                this.openInfoWindow()
            })), i
        }

        var ze = {
                71e4: {cp: [121.509062, 25.044332], name: "台湾"},
                13e4: {cp: [114.502461, 38.045474], name: "河北"},
                14e4: {cp: [112.549248, 37.857014], name: "山西"},
                15e4: {cp: [111.670801, 40.818311], name: "内蒙古"},
                21e4: {cp: [123.429096, 41.796767], name: "辽宁"},
                22e4: {cp: [125.3245, 43.886841], name: "吉林"},
                23e4: {cp: [126.642464, 45.756967], name: "黑龙江"},
                32e4: {cp: [118.767413, 32.041544], name: "江苏"},
                33e4: {cp: [120.153576, 30.287459], name: "浙江"},
                34e4: {cp: [117.283042, 31.86119], name: "安徽"},
                35e4: {cp: [119.306239, 26.075302], name: "福建"},
                36e4: {cp: [115.892151, 28.676493], name: "江西"},
                37e4: {cp: [117.000923, 36.675807], name: "山东"},
                41e4: {cp: [113.665412, 34.757975], name: "河南"},
                42e4: {cp: [114.298572, 30.584355], name: "湖北"},
                43e4: {cp: [112.982279, 28.19409], name: "湖南"},
                44e4: {cp: [113.280637, 23.125178], name: "广东"},
                45e4: {cp: [108.320004, 22.82402], name: "广西"},
                46e4: {cp: [110.33119, 20.031971], name: "海南"},
                51e4: {cp: [104.065735, 30.659462], name: "四川"},
                52e4: {cp: [106.713478, 26.578343], name: "贵州"},
                53e4: {cp: [102.712251, 25.040609], name: "云南"},
                54e4: {cp: [91.132212, 29.660361], name: "西藏"},
                61e4: {cp: [108.948024, 34.263161], name: "陕西"},
                62e4: {cp: [103.823557, 36.058039], name: "甘肃"},
                63e4: {cp: [101.778916, 36.623178], name: "青海"},
                64e4: {cp: [106.278179, 38.46637], name: "宁夏"},
                65e4: {cp: [87.617733, 43.792818], name: "新疆"},
                11e4: {cp: [116.405285, 39.904989], name: "北京"},
                12e4: {cp: [117.190182, 39.125596], name: "天津"},
                31e4: {cp: [121.472644, 31.231706], name: "上海"},
                5e5: {cp: [106.504962, 29.533155], name: "重庆"},
                81e4: {cp: [114.173355, 22.320048], name: "香港"},
                82e4: {cp: [113.54909, 22.198951], name: "澳门"}
            }, je = (a("6062"), a("3ca3"), a("ddb0"), a("d4ec")), _e = a("bee2"), Ue = a("99de"), He = a("7e84"),
            Ne = a("262e");
        a("a434");
        Ae["EffectComposer"] = function (e, t) {
            if (this.renderer = e, void 0 === t) {
                var a = {minFilter: Ae["s"], magFilter: Ae["s"], format: Ae["I"], stencilBuffer: !1},
                    r = e.getDrawingBufferSize(new Ae["T"]);
                t = new Ae["W"](r.width, r.height, a), t.texture.name = "EffectComposer.rt1"
            }
            this.renderTarget1 = t, this.renderTarget2 = t.clone(), this.renderTarget2.texture.name = "EffectComposer.rt2", this.writeBuffer = this.renderTarget1, this.readBuffer = this.renderTarget2, this.renderToScreen = !0, this.passes = [], void 0 === Ae["CopyShader"] && console.error("THREE.EffectComposer relies on THREE.CopyShader"), void 0 === Ae["ShaderPass"] && console.error("THREE.EffectComposer relies on THREE.ShaderPass"), this.copyPass = new Ae["ShaderPass"](Ae["CopyShader"]), this._previousFrameTime = Date.now()
        }, Object.assign(Ae["EffectComposer"].prototype, {
            swapBuffers: function () {
                var e = this.readBuffer;
                this.readBuffer = this.writeBuffer, this.writeBuffer = e
            }, addPass: function (e) {
                this.passes.push(e);
                var t = this.renderer.getDrawingBufferSize(new Ae["T"]);
                e.setSize(t.width, t.height)
            }, insertPass: function (e, t) {
                this.passes.splice(t, 0, e)
            }, isLastEnabledPass: function (e) {
                for (var t = e + 1; t < this.passes.length; t++) if (this.passes[t].enabled) return !1;
                return !0
            }, render: function (e) {
                void 0 === e && (e = .001 * (Date.now() - this._previousFrameTime)), this._previousFrameTime = Date.now();
                var t, a, r = this.renderer.getRenderTarget(), i = !1, n = this.passes.length;
                for (a = 0; a < n; a++) if (t = this.passes[a], !1 !== t.enabled) {
                    if (t.renderToScreen = this.renderToScreen && this.isLastEnabledPass(a), t.render(this.renderer, this.writeBuffer, this.readBuffer, e, i), t.needsSwap) {
                        if (i) {
                            var o = this.renderer.context;
                            o.stencilFunc(o.NOTEQUAL, 1, 4294967295), this.copyPass.render(this.renderer, this.writeBuffer, this.readBuffer, e), o.stencilFunc(o.EQUAL, 1, 4294967295)
                        }
                        this.swapBuffers()
                    }
                    void 0 !== Ae["MaskPass"] && (t instanceof Ae["MaskPass"] ? i = !0 : t instanceof Ae["ClearMaskPass"] && (i = !1))
                }
                this.renderer.setRenderTarget(r)
            }, reset: function (e) {
                if (void 0 === e) {
                    var t = this.renderer.getDrawingBufferSize(new Ae["T"]);
                    e = this.renderTarget1.clone(), e.setSize(t.width, t.height)
                }
                this.renderTarget1.dispose(), this.renderTarget2.dispose(), this.renderTarget1 = e, this.renderTarget2 = e.clone(), this.writeBuffer = this.renderTarget1, this.readBuffer = this.renderTarget2
            }, setSize: function (e, t) {
                this.renderTarget1.setSize(e, t), this.renderTarget2.setSize(e, t);
                for (var a = 0; a < this.passes.length; a++) this.passes[a].setSize(e, t)
            }
        }), Ae["Pass"] = function () {
            this.enabled = !0, this.needsSwap = !0, this.clear = !1, this.renderToScreen = !1
        }, Object.assign(Ae["Pass"].prototype, {
            setSize: function (e, t) {
            }, render: function (e, t, a, r, i) {
                console.error("THREE.Pass: .render() must be implemented in derived pass.")
            }
        }), Ae["Pass"].FullScreenQuad = function () {
            var e = new Ae["B"](-1, 1, 1, -1, 0, 1), t = new Ae["E"](2, 2), a = function (e) {
                this._mesh = new Ae["x"](t, e)
            };
            return Object.defineProperty(a.prototype, "material", {
                get: function () {
                    return this._mesh.material
                }, set: function (e) {
                    this._mesh.material = e
                }
            }), Object.assign(a.prototype, {
                render: function (t) {
                    t.render(this._mesh, e)
                }
            }), a
        }();
        Ae["Pass"];
        var Le = Ae["EffectComposer"];
        Ae["RenderPass"] = function (e, t, a, r, i) {
            Ae["Pass"].call(this), this.scene = e, this.camera = t, this.overrideMaterial = a, this.clearColor = r, this.clearAlpha = void 0 !== i ? i : 0, this.clear = !0, this.clearDepth = !1, this.needsSwap = !1
        }, Ae["RenderPass"].prototype = Object.assign(Object.create(Ae["Pass"].prototype), {
            constructor: Ae["RenderPass"],
            render: function (e, t, a, r, i) {
                var n, o, s = e.autoClear;
                e.autoClear = !1, this.scene.overrideMaterial = this.overrideMaterial, this.clearColor && (n = e.getClearColor().getHex(), o = e.getClearAlpha(), e.setClearColor(this.clearColor, this.clearAlpha)), this.clearDepth && e.clearDepth(), e.setRenderTarget(this.renderToScreen ? null : a), this.clear && e.clear(e.autoClearColor, e.autoClearDepth, e.autoClearStencil), e.render(this.scene, this.camera), this.clearColor && e.setClearColor(n, o), this.scene.overrideMaterial = null, e.autoClear = s
            }
        });
        var Ge = Ae["RenderPass"], We = Ge;
        Ae["CopyShader"] = {
            uniforms: {tDiffuse: {value: null}, opacity: {value: 1}},
            vertexShader: ["varying vec2 vUv;", "void main() {", "vUv = uv;", "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );", "}"].join("\n"),
            fragmentShader: ["uniform float opacity;", "uniform sampler2D tDiffuse;", "varying vec2 vUv;", "void main() {", "vec4 texel = texture2D( tDiffuse, vUv );", "gl_FragColor = opacity * texel;", "}"].join("\n")
        };
        Ae["CopyShader"];
        Ae["ShaderPass"] = function (e, t) {
            Ae["Pass"].call(this), this.textureID = void 0 !== t ? t : "tDiffuse", e instanceof Ae["M"] ? (this.uniforms = e.uniforms, this.material = e) : e && (this.uniforms = Ae["R"].clone(e.uniforms), this.material = new Ae["M"]({
                defines: Object.assign({}, e.defines),
                uniforms: this.uniforms,
                vertexShader: e.vertexShader,
                fragmentShader: e.fragmentShader
            })), this.fsQuad = new Ae["Pass"].FullScreenQuad(this.material)
        }, Ae["ShaderPass"].prototype = Object.assign(Object.create(Ae["Pass"].prototype), {
            constructor: Ae["ShaderPass"],
            render: function (e, t, a, r, i) {
                this.uniforms[this.textureID] && (this.uniforms[this.textureID].value = a.texture), this.fsQuad.material = this.material, this.renderToScreen ? (e.setRenderTarget(null), this.fsQuad.render(e)) : (e.setRenderTarget(t), this.clear && e.clear(e.autoClearColor, e.autoClearDepth, e.autoClearStencil), this.fsQuad.render(e))
            }
        });
        Ae["ShaderPass"];
        Ae["LuminosityHighPassShader"] = {
            shaderID: "luminosityHighPass",
            uniforms: {
                tDiffuse: {value: null},
                luminosityThreshold: {value: 1},
                smoothWidth: {value: 1},
                defaultColor: {value: new Ae["f"](0)},
                defaultOpacity: {value: 0}
            },
            vertexShader: ["varying vec2 vUv;", "void main() {", "vUv = uv;", "gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );", "}"].join("\n"),
            fragmentShader: ["uniform sampler2D tDiffuse;", "uniform vec3 defaultColor;", "uniform float defaultOpacity;", "uniform float luminosityThreshold;", "uniform float smoothWidth;", "varying vec2 vUv;", "void main() {", "vec4 texel = texture2D( tDiffuse, vUv );", "vec3 luma = vec3( 0.299, 0.587, 0.114 );", "float v = dot( texel.xyz, luma );", "vec4 outputColor = vec4( defaultColor.rgb, defaultOpacity );", "float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );", "gl_FragColor = mix( outputColor, texel, alpha );", "}"].join("\n")
        };
        Ae["LuminosityHighPassShader"];
        Ae["UnrealBloomPass"] = function (e, t, a, r) {
            Ae["Pass"].call(this), this.strength = void 0 !== t ? t : 1, this.radius = a, this.threshold = r, this.resolution = void 0 !== e ? new Ae["T"](e.x, e.y) : new Ae["T"](256, 256), this.clearColor = new Ae["f"](0, 0, 0);
            var i = {minFilter: Ae["s"], magFilter: Ae["s"], format: Ae["I"]};
            this.renderTargetsHorizontal = [], this.renderTargetsVertical = [], this.nMips = 5;
            var n = Math.round(this.resolution.x / 2), o = Math.round(this.resolution.y / 2);
            this.renderTargetBright = new Ae["W"](n, o, i), this.renderTargetBright.texture.name = "UnrealBloomPass.bright", this.renderTargetBright.texture.generateMipmaps = !1;
            for (var s = 0; s < this.nMips; s++) {
                var l = new Ae["W"](n, o, i);
                l.texture.name = "UnrealBloomPass.h" + s, l.texture.generateMipmaps = !1, this.renderTargetsHorizontal.push(l);
                var u = new Ae["W"](n, o, i);
                u.texture.name = "UnrealBloomPass.v" + s, u.texture.generateMipmaps = !1, this.renderTargetsVertical.push(u), n = Math.round(n / 2), o = Math.round(o / 2)
            }
            void 0 === Ae["LuminosityHighPassShader"] && console.error("THREE.UnrealBloomPass relies on THREE.LuminosityHighPassShader");
            var c = Ae["LuminosityHighPassShader"];
            this.highPassUniforms = Ae["R"].clone(c.uniforms), this.highPassUniforms["luminosityThreshold"].value = r, this.highPassUniforms["smoothWidth"].value = .01, this.materialHighPassFilter = new Ae["M"]({
                uniforms: this.highPassUniforms,
                vertexShader: c.vertexShader,
                fragmentShader: c.fragmentShader,
                defines: {}
            }), this.separableBlurMaterials = [];
            var d = [3, 5, 7, 9, 11];
            n = Math.round(this.resolution.x / 2), o = Math.round(this.resolution.y / 2);
            for (var h = 0; h < this.nMips; h++) this.separableBlurMaterials.push(this.getSeperableBlurMaterial(d[h])), this.separableBlurMaterials[h].uniforms["texSize"].value = new Ae["T"](n, o), n = Math.round(n / 2), o = Math.round(o / 2);
            this.compositeMaterial = this.getCompositeMaterial(this.nMips), this.compositeMaterial.uniforms["blurTexture1"].value = this.renderTargetsVertical[0].texture, this.compositeMaterial.uniforms["blurTexture2"].value = this.renderTargetsVertical[1].texture, this.compositeMaterial.uniforms["blurTexture3"].value = this.renderTargetsVertical[2].texture, this.compositeMaterial.uniforms["blurTexture4"].value = this.renderTargetsVertical[3].texture, this.compositeMaterial.uniforms["blurTexture5"].value = this.renderTargetsVertical[4].texture, this.compositeMaterial.uniforms["bloomStrength"].value = t, this.compositeMaterial.uniforms["bloomRadius"].value = .1, this.compositeMaterial.needsUpdate = !0;
            var m = [1, .8, .6, .4, .2];
            this.compositeMaterial.uniforms["bloomFactors"].value = m, this.bloomTintColors = [new Ae["U"](1, 1, 1), new Ae["U"](1, 1, 1), new Ae["U"](1, 1, 1), new Ae["U"](1, 1, 1), new Ae["U"](1, 1, 1)], this.compositeMaterial.uniforms["bloomTintColors"].value = this.bloomTintColors, void 0 === Ae["CopyShader"] && console.error("THREE.BloomPass relies on THREE.CopyShader");
            var p = Ae["CopyShader"];
            this.copyUniforms = Ae["R"].clone(p.uniforms), this.copyUniforms["opacity"].value = 1, this.materialCopy = new Ae["M"]({
                uniforms: this.copyUniforms,
                vertexShader: p.vertexShader,
                fragmentShader: p.fragmentShader,
                blending: Ae["a"],
                depthTest: !1,
                depthWrite: !1,
                transparent: !0
            }), this.enabled = !0, this.needsSwap = !1, this.oldClearColor = new Ae["f"], this.oldClearAlpha = 1, this.basic = new Ae["y"], this.fsQuad = new Ae["Pass"].FullScreenQuad(null)
        }, Ae["UnrealBloomPass"].prototype = Object.assign(Object.create(Ae["Pass"].prototype), {
            constructor: Ae["UnrealBloomPass"], dispose: function () {
                for (var e = 0; e < this.renderTargetsHorizontal.length; e++) this.renderTargetsHorizontal[e].dispose();
                for (var t = 0; t < this.renderTargetsVertical.length; t++) this.renderTargetsVertical[t].dispose();
                this.renderTargetBright.dispose()
            }, setSize: function (e, t) {
                var a = Math.round(e / 2), r = Math.round(t / 2);
                this.renderTargetBright.setSize(a, r);
                for (var i = 0; i < this.nMips; i++) this.renderTargetsHorizontal[i].setSize(a, r), this.renderTargetsVertical[i].setSize(a, r), this.separableBlurMaterials[i].uniforms["texSize"].value = new Ae["T"](a, r), a = Math.round(a / 2), r = Math.round(r / 2)
            }, render: function (e, t, a, r, i) {
                this.oldClearColor.copy(e.getClearColor()), this.oldClearAlpha = e.getClearAlpha();
                var n = e.autoClear;
                e.autoClear = !1, e.setClearColor(this.clearColor, 0), i && e.context.disable(e.context.STENCIL_TEST), this.renderToScreen && (this.fsQuad.material = this.basic, this.basic.map = a.texture, e.setRenderTarget(null), e.clear(), this.fsQuad.render(e)), this.highPassUniforms["tDiffuse"].value = a.texture, this.highPassUniforms["luminosityThreshold"].value = this.threshold, this.fsQuad.material = this.materialHighPassFilter, e.setRenderTarget(this.renderTargetBright), e.clear(), this.fsQuad.render(e);
                for (var o = this.renderTargetBright, s = 0; s < this.nMips; s++) this.fsQuad.material = this.separableBlurMaterials[s], this.separableBlurMaterials[s].uniforms["colorTexture"].value = o.texture, this.separableBlurMaterials[s].uniforms["direction"].value = Ae["UnrealBloomPass"].BlurDirectionX, e.setRenderTarget(this.renderTargetsHorizontal[s]), e.clear(), this.fsQuad.render(e), this.separableBlurMaterials[s].uniforms["colorTexture"].value = this.renderTargetsHorizontal[s].texture, this.separableBlurMaterials[s].uniforms["direction"].value = Ae["UnrealBloomPass"].BlurDirectionY, e.setRenderTarget(this.renderTargetsVertical[s]), e.clear(), this.fsQuad.render(e), o = this.renderTargetsVertical[s];
                this.fsQuad.material = this.compositeMaterial, this.compositeMaterial.uniforms["bloomStrength"].value = this.strength, this.compositeMaterial.uniforms["bloomRadius"].value = this.radius, this.compositeMaterial.uniforms["bloomTintColors"].value = this.bloomTintColors, e.setRenderTarget(this.renderTargetsHorizontal[0]), e.clear(), this.fsQuad.render(e), this.fsQuad.material = this.materialCopy, this.copyUniforms["tDiffuse"].value = this.renderTargetsHorizontal[0].texture, i && e.context.enable(e.context.STENCIL_TEST), this.renderToScreen ? (e.setRenderTarget(null), this.fsQuad.render(e)) : (e.setRenderTarget(a), this.fsQuad.render(e)), e.setClearColor(this.oldClearColor, this.oldClearAlpha), e.autoClear = n
            }, getSeperableBlurMaterial: function (e) {
                return new Ae["M"]({
                    defines: {KERNEL_RADIUS: e, SIGMA: e},
                    uniforms: {
                        colorTexture: {value: null},
                        texSize: {value: new Ae["T"](.5, .5)},
                        direction: {value: new Ae["T"](.5, .5)}
                    },
                    vertexShader: "letying vec2 vUv;\n      void main() {\n      vUv = uv;\n      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n      }",
                    fragmentShader: "#include <common>      letying vec2 vUv;\n      uniform sampler2D colorTexture;\n      uniform vec2 texSize;      uniform vec2 direction;            float gaussianPdf(in float x, in float sigma) {      return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;      }      void main() {\n      vec2 invSize = 1.0 / texSize;      float fSigma = float(SIGMA);      float weightSum = gaussianPdf(0.0, fSigma);      float alphaSum = 0.0;      vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;      for( int i = 1; i < KERNEL_RADIUS; i ++ ) {        float x = float(i);        float w = gaussianPdf(x, fSigma);        vec2 uvOffset = direction * invSize * x;        vec4 sample1 = texture2D( colorTexture, vUv + uvOffset);        vec4 sample2 = texture2D( colorTexture, vUv - uvOffset);        diffuseSum += (sample1.rgb + sample2.rgb) * w;        alphaSum += (sample1.a + sample2.a) * w;        weightSum += 2.0 * w;      }      gl_FragColor = vec4(diffuseSum/weightSum, alphaSum/weightSum);\n      }"
                })
            }, getCompositeMaterial: function (e) {
                return new Ae["M"]({
                    defines: {NUM_MIPS: e},
                    uniforms: {
                        blurTexture1: {value: null},
                        blurTexture2: {value: null},
                        blurTexture3: {value: null},
                        blurTexture4: {value: null},
                        blurTexture5: {value: null},
                        dirtTexture: {value: null},
                        bloomStrength: {value: 1},
                        bloomFactors: {value: null},
                        bloomTintColors: {value: null},
                        bloomRadius: {value: 0}
                    },
                    vertexShader: "letying vec2 vUv;\n      void main() {\n      vUv = uv;\n      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n      }",
                    fragmentShader: "letying vec2 vUv;      uniform sampler2D blurTexture1;      uniform sampler2D blurTexture2;      uniform sampler2D blurTexture3;      uniform sampler2D blurTexture4;      uniform sampler2D blurTexture5;      uniform sampler2D dirtTexture;      uniform float bloomStrength;      uniform float bloomRadius;      uniform float bloomFactors[NUM_MIPS];      uniform vec3 bloomTintColors[NUM_MIPS];            float lerpBloomFactor(const in float factor) {       float mirrorFactor = 1.2 - factor;      return mix(factor, mirrorFactor, bloomRadius);      }            void main() {      gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) +                lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) +                lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) +                lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) +                lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );      }"
                })
            }
        }), Ae["UnrealBloomPass"].BlurDirectionX = new Ae["T"](1, 0), Ae["UnrealBloomPass"].BlurDirectionY = new Ae["T"](0, 1);
        var Ve, Qe = Ae["UnrealBloomPass"], Ye = Qe;

        function Xe() {
            var e = {exposure: 1, bloomStrength: 4.5, bloomThreshold: 0, bloomRadius: 0, debug: !1},
                t = this.getThreeRenderer(), a = this.getMap().getSize();
            this.composer = new Le(t), this.composer.setSize(a.width, a.height);
            var r = this.getScene(), i = this.getCamera();
            this.renderPass = new We(r, i), this.composer.addPass(this.renderPass);
            var n = this.bloomPass = new Ye(new Ae["T"](a.width, a.height));
            n.renderToScreen = !0, n.threshold = e.bloomThreshold, n.strength = e.bloomStrength, n.radius = e.bloomRadius, this.composer.addPass(n), this.bloomEnable = !0
        }

        function qe() {
            this.getRenderer().renderScene = function () {
                var e = this.layer;
                e._callbackBaseObjectAnimation(), this._syncCamera();
                var t = this.context, a = this.camera, r = this.scene;
                e.bloomEnable && e.composer && e.composer.passes.length > 1 ? (t.autoClear && (t.autoClear = !1), e.bloomPass && a.layers.set(1), e && e.composer && e.composer.render(0), t.clearDepth(), a.layers.set(0), t.render(r, a)) : (t.autoClear || (t.autoClear = !0), t.render(r, a)), this.completeRender()
            }
        }

        function Ke() {
            Pe["b"].prototype.initBloom = Xe, Pe["b"].prototype.setRendererRenderScene = qe
        }

        function Je(e, t) {
            var a = [];
            e.getLayer("lines") && Ve.remove(), Ve = new Pe["b"]("lines", {
                forceRenderOnMoving: !0,
                forceRenderOnRotating: !0
            }), Ve.prepareToDraw = function (e, r) {
                var i = new Ae["i"](16777215);
                i.position.set(0, -10, 10).normalize(), r.add(i), this.initBloom(), this.setRendererRenderScene(), t.features.forEach((function (e) {
                    var t = e.properties.name;
                    t.length < 5 && (t += "   ");
                    var r = tt(50, t), i = new rt(e.properties.cp, {len: t.length}, r, Ve);
                    a.push(i)
                })), Ve.addMesh(a), $e(t)
            }, Ve.addTo(e), Ke()
        }

        function Ze(e) {
            var t = {features: [], type: "FeatureCollection"};
            return e.features.forEach((function (e) {
                if ("MultiPolygon" != e.geometry.type) {
                    var a = [], r = new Set;
                    e.geometry.coordinates[0].forEach((function (e, t) {
                        r.has(e.join("_")) || (t > 0 && r.add(e.join("_")), a.push(e))
                    }));
                    var i = {
                        type: "Feature",
                        properties: {name: e.properties.name, date: e.properties.date},
                        geometry: {type: "LineString", coordinates: a}
                    };
                    t.features.push(i)
                }
            })), t
        }

        function $e(e) {
            var t = new Ae["q"]({linewidth: 1, color: "rgb(255,90,0)", blending: Ae["a"], transparent: !0}), a = Ze(e),
                r = Me["e"].toGeometry(a), i = r.map((function (e) {
                    var a = Ve.toLine(e, {}, t);
                    return a.getObject3d().layers.enable(1), a
                }));
            Ve.addMesh(i), et(r)
        }

        function et(e) {
            var t = new Ae["y"]({color: "rgb(255,45,0)", transparent: !0, blending: Ae["a"]}),
                a = new Ae["y"]({color: "#ffffff", transparent: !0}), r = [];
            e.forEach((function (e) {
                var a = Ve.toExtrudeLine(e, {altitude: 0, width: 10, height: 50}, t);
                a.getObject3d().layers.enable(1), r.push(a)
            }));
            var i = [];
            e.forEach((function (e) {
                var t = Ve.toExtrudeLineTrail(e, {
                    altitude: 0,
                    width: 10,
                    height: 30,
                    chunkLength: 50,
                    speed: 1,
                    trail: 6
                }, a);
                t.getObject3d().layers.enable(1), i.push(t)
            })), Ve.addMesh(r), Ve.addMesh(i), it()
        }

        function tt(e, t) {
            var a = 400, r = document.createElement("canvas");
            r.width = a, r.height = 16 * t.length;
            var i = r.getContext("2d"), n = i.createLinearGradient(0, 0, a, 0);
            n.addColorStop("0.0", "#ff0000"), n.addColorStop("1.0", "#ff0000"), i.strokeStyle = n, i.font = "".concat(e, "px Aria"), i.textAlign = "left", i.textBaseline = "top", i.fillStyle = "#f00", i.fillText(t, 5, 10);
            var o = new Ae["P"](r);
            o.needsUpdate = !0;
            var s = new Ae["z"]({map: o, side: Ae["m"], transparent: !1});
            return s
        }

        var at = {len: 4, altitude: 0}, rt = function (e) {
            function t(e, a, r, i) {
                var n;
                Object(je["a"])(this, t), a = Me["l"].extend({}, at, a, {
                    layer: i,
                    coordinate: e
                }), n = Object(Ue["a"])(this, Object(He["a"])(t).call(this)), n._initOptions(a);
                var o = new Ae["F"](1.5 * a.len, 4.5);
                n._createMesh(o, r);
                var s = i.coordinateToVector3(e, 0);
                return n.getObject3d().position.copy(s), n
            }

            return Object(Ne["a"])(t, e), Object(_e["a"])(t, [{
                key: "animateShow", value: function () {
                    var e = this, t = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
                        a = arguments.length > 1 ? arguments[1] : void 0;
                    this._showPlayer && this._showPlayer.cancel(), Me["l"].isFunction(t) && (t = {}, a = t);
                    var r = t["duration"] || 1e3, i = t["easing"] || "out",
                        n = this._showPlayer = Me["n"].Animation.animate({scale: 1}, {
                            duration: r,
                            easing: i
                        }, (function (t) {
                            var r = t.styles.scale;
                            r > 0 && e.getObject3d().scale.set(r, r, r), a && a(t, r)
                        }));
                    return n.play(), n
                }
            }]), t
        }(Pe["a"]);

        function it() {
            Ve._needsUpdate = !Ve._needsUpdate, Ve._needsUpdate && (Ve.getRenderer().clearCanvas(), Ve.renderScene()), requestAnimationFrame(it)
        }

        var nt = {
            name: "Map", data: function () {
                return {
                    map: null,
                    planeLayer: null,
                    buildingLayer: null,
                    currentProvince: "510000",
                    currentCity: "",
                    data: null,
                    maps: [{label: "常规地图", map: "Gaode", type: "0", index: 0}, {
                        label: "暗黑2.5D",
                        map: "Gaode",
                        type: "1",
                        index: 1
                    }, {label: "3D场景", map: "Gaode", type: "1", index: 2}],
                    mapIndex: 1,
                    provinces: Object.keys(ze).map((function (e) {
                        return {code: e, name: ze[e].name}
                    }))
                }
            }, mounted: function () {
                this.mapIndex = parseInt(F.getCookie("mapIndex", 1)), this.init(), this.loadProvince(this.currentProvince)
            }, methods: {
                init: function () {
                    var e = Ee.getConfig();
                    this.map = new Me["h"]("map", e), this.map.on("moveend", Ee.handleMapMove)
                }, loadProvince: function (e) {
                    var t = this;
                    Ee.loadData(e, (function (a) {
                        var r = e in ze ? ze[e].cp : null;
                        a.features[0] && (r = a.features[0].properties.cp), t.map.setCenter(r), t.map.setZoom(11), t.data = a, t.loadMap(t.mapIndex, a)
                    }))
                }, loadMap: function (e, t) {
                    var a = [Re, De, Je];
                    a[e](this.map, t)
                }, resetMap: function (e) {
                    var t = this.maps[e];
                    F.setCookie("mapStyle", [t.map, t.type].join("_")), F.setCookie("mapIndex", e), this.map.remove(), this.map = null, this.init(), this.loadMap(e, this.data)
                }
            }
        }, ot = nt, st = (a("5490"), Object(c["a"])(ot, be, ye, !1, null, null, null)), lt = st.exports;
        r["default"].use(S["a"]);
        var ut = [{path: "", component: xe}, {path: "/china", component: xe}, {
            path: "/province",
            component: xe
        }, {path: "/map", component: lt}], ct = new S["a"]({routes: ut}), dt = ct;
        r["default"].prototype.$echarts = x.a, r["default"].use(y.a), r["default"].config.productionTip = !1, new r["default"]({
            el: "#app",
            router: dt,
            render: function (e) {
                return e(v)
            }
        })
    }, a263: function (e, t, a) {
    }, a6b6: function (e, t, a) {
    }, b9de: function (e, t, a) {
    }, cc5b: function (e, t, a) {
    }, e4bd: function (e, t, a) {
        "use strict";
        var r = a("cc5b"), i = a.n(r);
        i.a
    }
});
//# sourceMappingURL=index.18d25166.js.map