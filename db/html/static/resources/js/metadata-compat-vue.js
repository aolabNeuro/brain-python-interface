/**
 * Metadata compatibility module
 * Preserves the Metadata API used by list-vue.js
 */

const metadataCompatRoot = typeof window !== 'undefined' ? window : globalThis;

function Metadata() {
    this.useVue = !!(window.metadataApp && window.metadataApp.instance);
    if (this.useVue) {
        return;
    }

    $("#metadata_table").html("");
    var params = new Parameters(editable=true);
    this.params = params;
    $("#metadata_table").append(this.params.obj);
    var add_new_row = $('<input class="paramadd" type="button" value="+"/>');
    add_new_row.on("click", function() {params.add_row();});
    this.add_new_row = add_new_row;
    $("#metadata_table").append(add_new_row);
}

Metadata.prototype.update = function(info) {
    if (this.useVue && window.metadataApp && window.metadataApp.instance) {
        window.metadataApp.instance.updateMetadata(info);
        return;
    }
    this.params.update(info);
};

Metadata.prototype.enable = function() {
    if (this.useVue && window.metadataApp && window.metadataApp.instance) {
        window.metadataApp.instance.setEditMode(true);
        return;
    }
    this.params.enable();
    this.add_new_row.show();
};

Metadata.prototype.disable = function() {
    if (this.useVue && window.metadataApp && window.metadataApp.instance) {
        window.metadataApp.instance.setEditMode(false);
        return;
    }
    this.params.disable();
    this.add_new_row.hide();
};

Metadata.prototype.get_data = function () {
    if (this.useVue && window.metadataApp && window.metadataApp.instance) {
        return window.metadataApp.instance.getMetadataValues();
    }
    var data = this.params.to_json();
    return data;
};

Metadata.prototype.reset = function () {
    if (this.useVue && window.metadataApp && window.metadataApp.instance) {
        window.metadataApp.instance.updateMetadata(window.metadataApp.instance.metadata || {});
        return;
    }
    this.params.clear_all();
};

metadataCompatRoot.Metadata = Metadata;

if (typeof(module) !== 'undefined' && module.exports) {
    exports.Metadata = Metadata;
    exports.$ = (typeof $ !== 'undefined') ? $ : undefined;
}