string=$( YLightSensor any get_current_value )
cur_lux=$(echo "$string" | grep -oE '[0-9]+\.[0-9]+' | tail -1)

echo Lux: $cur_lux

