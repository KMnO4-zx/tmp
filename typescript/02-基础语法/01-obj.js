var Site = /** @class */ (function () {
    function Site() {
    }
    Site.prototype.name = function () {
        console.log("我是不要葱姜蒜");
    };
    return Site;
}());
var obj = new Site();
obj.name();
