from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponseRedirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('visualize/', include('visualizer.urls')),  # visualizer 앱의 URLs 포함
    path('', lambda request: HttpResponseRedirect('/visualize/')),  # 루트 URL로 접속 시 /visualize/로 리디렉션
]
