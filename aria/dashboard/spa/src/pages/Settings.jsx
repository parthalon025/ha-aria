import { useState, useEffect, useRef } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';
import LoadingState from '../components/LoadingState.jsx';
import ErrorState from '../components/ErrorState.jsx';

/** Group configs by category for collapsible sections. */
function groupByCategory(configs) {
  const groups = {};
  for (const c of configs) {
    const cat = c.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(c);
  }
  return groups;
}

function ParamControl({ config, onUpdate }) {
  const [value, setValue] = useState(config.value);
  const [saving, setSaving] = useState(false);
  const timerRef = useRef(null);
  const isDefault = config.value === config.default_value;

  useEffect(() => { setValue(config.value); }, [config.value]);

  function debouncedSave(newVal) {
    setValue(newVal);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      setSaving(true);
      try {
        await putJson(`/api/config/${config.key}`, { value: String(newVal) });
        onUpdate();
      } catch (e) {
        console.error('Config save failed:', e);
      } finally {
        setSaving(false);
      }
    }, 500);
  }

  async function handleReset() {
    setSaving(true);
    try {
      await postJson(`/api/config/reset/${config.key}`, {});
      setValue(config.default_value);
      onUpdate();
    } catch (e) {
      console.error('Config reset failed:', e);
    } finally {
      setSaving(false);
    }
  }

  const vtype = config.value_type;

  return (
    <div class="flex flex-col sm:flex-row sm:items-center gap-2 py-3 border-b border-gray-100 last:border-0">
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2">
          <span class="text-sm font-medium text-gray-800">{config.label || config.key}</span>
          {saving && <span class="text-xs text-blue-500">Saving...</span>}
        </div>
        {config.description && (
          <p class="text-xs text-gray-400 mt-0.5">{config.description}</p>
        )}
      </div>
      <div class="flex items-center gap-2 flex-shrink-0">
        {vtype === 'number' && (
          <div class="flex items-center gap-2">
            <input
              type="range"
              min={config.min_value ?? 0}
              max={config.max_value ?? 100}
              step={config.step ?? 1}
              value={value}
              onInput={(e) => debouncedSave(e.target.value)}
              class="w-28 accent-blue-500"
            />
            <span class="text-sm font-mono text-gray-700 w-16 text-right">{value}</span>
          </div>
        )}
        {vtype === 'boolean' && (
          <button
            class={`relative w-10 h-5 rounded-full transition-colors ${
              value === 'true' || value === '1' ? 'bg-blue-500' : 'bg-gray-300'
            }`}
            onClick={() => debouncedSave(value === 'true' || value === '1' ? 'false' : 'true')}
          >
            <span class={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
              value === 'true' || value === '1' ? 'translate-x-5' : 'translate-x-0.5'
            }`} />
          </button>
        )}
        {vtype === 'string' && (
          <input
            type="text"
            value={value}
            onInput={(e) => debouncedSave(e.target.value)}
            class="border border-gray-300 rounded px-2 py-1 text-sm w-48 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        )}
        {vtype === 'select' && (
          <select
            value={value}
            onChange={(e) => debouncedSave(e.target.value)}
            class="border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            {(config.options || '').split(',').map((o) => (
              <option key={o.trim()} value={o.trim()}>{o.trim()}</option>
            ))}
          </select>
        )}
        <button
          onClick={handleReset}
          disabled={isDefault}
          class={`text-xs px-2 py-1 rounded ${
            isDefault
              ? 'text-gray-300 cursor-default'
              : 'text-blue-500 hover:bg-blue-50'
          }`}
          title="Reset to default"
        >
          Reset
        </button>
      </div>
    </div>
  );
}

function CategorySection({ category, configs, onUpdate }) {
  const [open, setOpen] = useState(true);

  return (
    <section class="bg-white rounded-lg shadow-sm">
      <button
        class="w-full flex items-center justify-between px-4 py-3 text-left"
        onClick={() => setOpen(!open)}
      >
        <h3 class="text-sm font-bold text-gray-700">{category}</h3>
        <span class="text-gray-400 text-xs">{open ? '\u25B2' : '\u25BC'} {configs.length} param{configs.length !== 1 ? 's' : ''}</span>
      </button>
      {open && (
        <div class="px-4 pb-3">
          {configs.map((c) => (
            <ParamControl key={c.key} config={c} onUpdate={onUpdate} />
          ))}
        </div>
      )}
    </section>
  );
}

export default function Settings() {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function fetchConfigs() {
    try {
      const data = await fetchJson('/api/config');
      setConfigs(data.configs || []);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchConfigs(); }, []);

  if (loading && configs.length === 0) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Settings</h1>
          <p class="text-sm text-gray-500">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <LoadingState type="full" />
      </div>
    );
  }

  if (error) {
    return (
      <div class="space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">Settings</h1>
          <p class="text-sm text-gray-500">Configure shadow engine, activity monitor, and data quality parameters.</p>
        </div>
        <ErrorState error={error} onRetry={fetchConfigs} />
      </div>
    );
  }

  const groups = groupByCategory(configs);
  const modified = configs.filter((c) => c.value !== c.default_value).length;

  return (
    <div class="space-y-6">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Settings</h1>
        <p class="text-sm text-gray-500">
          {configs.length} parameters across {Object.keys(groups).length} categories.
          {modified > 0 && <span class="text-blue-500 ml-1">{modified} modified from defaults.</span>}
        </p>
      </div>

      <div class="space-y-4">
        {Object.entries(groups).map(([cat, items]) => (
          <CategorySection key={cat} category={cat} configs={items} onUpdate={fetchConfigs} />
        ))}
      </div>
    </div>
  );
}
