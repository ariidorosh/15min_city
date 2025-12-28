/* map_bridge.js
 * QtWebChannel ↔ Folium map click-pick bridge
 * Підтримка: start / end / stop + category
 */

(function () {

  function resolveLeafletMap(passedMap) {
    try {
      if (passedMap && typeof passedMap.on === 'function') {
        return passedMap;
      }
    } catch (e) {}

    // 1) Найнадійніше: взяти id folium-container і спробувати window[id]
    try {
      var el = document.querySelector('.folium-map');
      if (el && el.id) {
        var m = window[el.id];
        if (m && typeof m.on === 'function') {
          console.log('[map_bridge] resolved map via .folium-map id:', el.id);
          return m;
        }
      }
    } catch (e) {}

    // 2) Фолбек: пошук по window.* (працює, якщо var map_* реально глобальний)
    try {
      for (var k in window) {
        if (!k || k.indexOf('map_') !== 0) continue;
        var v = window[k];
        if (v && typeof v.on === 'function') {
          console.log('[map_bridge] resolved map via window scan:', k);
          return v;
        }
      }
    } catch (e) {}

    return null;
  }

  function waitForMap() {
    if (typeof qt === 'undefined' || !qt.webChannelTransport) {
      console.log('[map_bridge] qt or webChannelTransport is undefined');
      setTimeout(waitForMap, 100);
      return;
    }

    if (typeof QWebChannel === 'undefined') {
      console.log('[map_bridge] QWebChannel not ready');
      setTimeout(waitForMap, 100);
      return;
    }

    if (typeof L === 'undefined') {
      console.log('[map_bridge] Leaflet (L) not ready');
      setTimeout(waitForMap, 100);
      return;
    }

    // Важливо: folium map може бути “не тим об’єктом”, тому резолвимо правильно
    var realMap = resolveLeafletMap(window.__folium_map__);
    if (!realMap) {
      console.log('[map_bridge] leaflet map instance not ready (resolve failed)');
      setTimeout(waitForMap, 100);
      return;
    }

    initBridge(realMap);
  }

  function initBridge(map) {
    if (window.__map_bridge_initialized) {
      console.log('[map_bridge] already initialized');
      return;
    }
    window.__map_bridge_initialized = true;

    window.__pick_target = window.__pick_target || '';
    window.__pick_category = window.__pick_category || '';
    window.__stopMarkers = window.__stopMarkers || [];
    window.__startMarker = window.__startMarker || null;
    window.__endMarker = window.__endMarker || null;

    try {
      new QWebChannel(qt.webChannelTransport, function (channel) {
        window.bridge = channel.objects.bridge;
        console.log('[map_bridge] WebChannel bridge ready');
      });
    } catch (e) {
      console.log('[map_bridge] QWebChannel init error', e);
      window.bridge = null;
    }

    function setMarker(kind, lat, lon) {
      try {
        if (kind === 'start') {
          if (window.__startMarker) map.removeLayer(window.__startMarker);
          window.__startMarker = L.marker([lat, lon]).addTo(map).bindPopup('Start');
          return;
        }

        if (kind === 'end') {
          if (window.__endMarker) map.removeLayer(window.__endMarker);
          window.__endMarker = L.marker([lat, lon]).addTo(map).bindPopup('End');
          return;
        }

        if (kind === 'stop') {
          var idx = (window.__stopMarkers ? window.__stopMarkers.length : 0) + 1;
          var mk = L.marker([lat, lon]).addTo(map).bindPopup('Stop #' + idx);
          window.__stopMarkers.push(mk);
          return;
        }
      } catch (e) {
        console.log('[map_bridge] marker error', e);
      }
    }

    function callBridge(lat, lon, target, category) {
      if (!window.bridge || !window.bridge.map_clicked) {
        console.log('[map_bridge] bridge not ready');
        return;
      }
      try {
        window.bridge.map_clicked(lat, lon, target || '', category || '');
      } catch (e) {
        console.log('[map_bridge] bridge call error', e);
      }
    }

    // Якщо тут map не Leaflet — ми би впали, але тепер resolveLeafletMap це гарантує
    map.on('click', function (e) {
      var tgt = (window.__pick_target || '').toLowerCase().trim();
      if (!tgt) return;

      if (tgt !== 'start' && tgt !== 'end' && tgt !== 'stop') {
        console.log('[map_bridge] invalid target:', tgt);
        return;
      }

      var lat = e.latlng.lat;
      var lon = e.latlng.lng;
      var cat = (window.__pick_category || '').toLowerCase().trim();

      setMarker(tgt, lat, lon);
      callBridge(lat, lon, tgt, cat);
    });

    console.log('[map_bridge] click handler attached');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', waitForMap);
  } else {
    waitForMap();
  }

})();
