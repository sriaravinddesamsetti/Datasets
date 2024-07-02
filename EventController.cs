using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Routing;
using MvcMovie.Models;

namespace MvcMovie.Controllers
{
    public class EventController : Controller
    {
        public ActionResult Index()
        {
            var eventBO = new EventBO();
            var events = eventBO.GetEvents();

            return View(events);
        }

        public ActionResult SortByName()
        {
            var eventBO = new EventBO();
            var events = eventBO.GetEvents();

            events.Sort((a, b) => a.Name.CompareTo(b.Name)); 

            return View(events);
        }
        public ActionResult SortByDate()
        {
            var eventBo = new EventBO();
            var events = eventBo.GetEvents();

            events.Sort((a, b) => a.Date.CompareTo(b.Date));

            return View(events);
        }
    }
}
