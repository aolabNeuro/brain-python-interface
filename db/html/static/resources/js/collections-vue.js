/**
 * Vue.js based Collections Component
 * Keeps compatibility with existing Collections class usage in list-vue.js
 */

const collectionsApp = {
    el: '#collections_vue',
    data() {
        return {
            collections: [],
            selectedCollections: {},
            isVisible: true,
        };
    },
    methods: {
        initializeCollections(collectionsList) {
            this.collections = collectionsList || [];
            const selected = {};
            this.collections.forEach((c) => {
                selected[c.name] = false;
            });
            this.selectedCollections = selected;
        },

        selectCollections(collections) {
            const selected = {};
            this.collections.forEach((c) => {
                selected[c.name] = false;
            });

            if (Array.isArray(collections)) {
                collections.forEach((name) => {
                    if (selected.hasOwnProperty(name)) {
                        selected[name] = true;
                    }
                });
            } else if (collections && typeof collections === 'object') {
                Object.keys(collections).forEach((name) => {
                    if (selected.hasOwnProperty(name)) {
                        selected[name] = !!collections[name];
                    }
                });
            }

            this.selectedCollections = selected;
        },

        setVisible(visible) {
            this.isVisible = !!visible;
        },
    },
    mounted() {
        console.log('Collections Vue app mounted');
    }
};

window.collectionsApp = collectionsApp;

function Collections() {
    this.collections_obj = $('#collections');
}

Collections.prototype.hide = function() {
    if (window.collectionsApp && window.collectionsApp.instance) {
        window.collectionsApp.instance.setVisible(false);
    }
    this.collections_obj.hide();
};

Collections.prototype.show = function() {
    if (window.collectionsApp && window.collectionsApp.instance) {
        window.collectionsApp.instance.setVisible(true);
    }
    this.collections_obj.show();
};

Collections.prototype.select_collections = function(collections) {
    if (window.collectionsApp && window.collectionsApp.instance) {
        window.collectionsApp.instance.selectCollections(collections);
        return;
    }
};

Collections.prototype.update_collection_membership = function(val) {
};

window.Collections = Collections;
