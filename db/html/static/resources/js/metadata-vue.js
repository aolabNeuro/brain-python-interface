/**
 * Vue.js based Metadata Component
 * Handles subject/experimenter/project/session and custom metadata fields
 */

const metadataApp = {
    el: '#metadata_vue',
    data() {
        return {
            metadata: {},
            editMode: true,
            showCustomRow: false,
            customKey: '',
            customValue: '',
        };
    },
    computed: {
        orderedMetadataEntries() {
            const preferredOrder = ['subject', 'experimenter', 'project', 'session'];
            const entries = Object.entries(this.metadata);
            entries.sort((a, b) => {
                const aIdx = preferredOrder.indexOf(a[0]);
                const bIdx = preferredOrder.indexOf(b[0]);
                const aRank = aIdx === -1 ? 999 : aIdx;
                const bRank = bIdx === -1 ? 999 : bIdx;
                if (aRank !== bRank) return aRank - bRank;
                return a[0].localeCompare(b[0]);
            });
            return entries;
        }
    },
    methods: {
        updateMetadata(desc) {
            if (!desc || typeof desc !== 'object') {
                this.metadata = {};
                return;
            }

            const normalized = {};
            Object.keys(desc).forEach((name) => {
                const field = { ...desc[name] };
                if (field.value === undefined || field.value === null) {
                    if (field.type === 'Bool') {
                        field.value = !!field.default;
                    } else if (field.default !== undefined) {
                        field.value = field.default;
                    } else {
                        field.value = '';
                    }
                }
                normalized[name] = field;
            });

            this.metadata = normalized;
            this.showCustomRow = false;
            this.customKey = '';
            this.customValue = '';
        },

        getMetadataValues() {
            const values = {};
            Object.keys(this.metadata).forEach((name) => {
                values[name] = this.metadata[name].value;
            });
            return values;
        },

        setEditMode(enabled) {
            this.editMode = enabled;
        },

        showAddCustomRow() {
            this.showCustomRow = true;
            this.customKey = '';
            this.customValue = '';
        },

        addCustomField() {
            const key = (this.customKey || '').trim();
            if (!key) return;
            if (this.metadata[key]) {
                this.showCustomRow = false;
                return;
            }
            this.metadata[key] = {
                type: 'String',
                default: '',
                desc: '',
                hidden: 'visible',
                value: this.customValue || '',
                required: false,
            };
            this.showCustomRow = false;
            this.customKey = '';
            this.customValue = '';
        },

        cancelCustomField() {
            this.showCustomRow = false;
            this.customKey = '';
            this.customValue = '';
        },

        removeCustomField(name) {
            if (['subject', 'experimenter', 'project', 'session'].includes(name)) return;
            delete this.metadata[name];
        },

        inputId(name) {
            return 'meta_' + name;
        },

        fieldLabel(name, field) {
            if (field && field.label) return field.label;
            return name;
        },

        fieldTitle(field) {
            return field && field.desc ? field.desc : '';
        },

        isType(field, type) {
            return field && field.type === type;
        },
    },
    mounted() {
        console.log('Metadata Vue app mounted');
    }
};

window.metadataApp = metadataApp;

window.updateVueMetadata = function(metadataDesc) {
    if (metadataApp.instance && metadataApp.instance.updateMetadata) {
        metadataApp.instance.updateMetadata(metadataDesc);
    }
};

window.getVueMetadataValues = function() {
    if (metadataApp.instance && metadataApp.instance.getMetadataValues) {
        return metadataApp.instance.getMetadataValues();
    }
    return {};
};

window.setVueMetadataEditMode = function(enabled) {
    if (metadataApp.instance && metadataApp.instance.setEditMode) {
        metadataApp.instance.setEditMode(enabled);
    }
};
