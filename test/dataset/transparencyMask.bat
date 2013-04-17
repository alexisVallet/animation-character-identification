for %%F in (.\*.png) do (
	convert %%F -channel Alpha -negate -separate %%F-mask.png
)